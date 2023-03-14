"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
#from mmflow.apis import inference_model, init_model
import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import face_alignment
from core.model_lm_talking import build_model
from core.checkpoint import CheckpointIO
from core.data_loader_lm_perceptual import InputFetcher
import core.utils_lm as utils
from metrics.eval import calculate_metrics
from tensorboardX import SummaryWriter

from ms1m_ir50.model_irse import IR_50
from scipy import spatial

from core.resnet50_ft_dims_2048 import resnet50_ft
from scipy.spatial.distance import cosine

import network
from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19
import FR_Pretrained_Test
from FR_Pretrained_Test.Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from core import VGG19_LOSS

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg19 = VGG19_LOSS.VGG19LOSS().to(device)

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)

        self.writer = SummaryWriter('log/test')

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_optims.ckpt'), **self.optims)]
        else:

            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'.format(args.resume_iter)), **self.nets_ema)]


        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):

        if self.args.loss == 'perceptual':
            criterion_id = network.LossEG(False, 0)

        if self.args.vggface2 :
            BACKBONE = resnet50_ft(weights_path='./FR_Pretrained_Test/Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
            print('VGGFace2 model built successfully')
            if torch.cuda.is_available():
                print("Use GPU Training")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            facemodel = BACKBONE.to(device)
        else:
            facemodel = None
        #####################################
        if self.args.landmark_loss :
            print("Loading the FAN Model......")

        #####################################
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, args.latent_dim, 'train',args.multi)
        fetcher_val = InputFetcher(loaders.val, args.latent_dim, 'val', args.multi)
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)



        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):

            inputs = next(fetcher)

            if args.multi:
                x1_1_source = inputs.x1
                x1_2_source = inputs.x2
                x1_3_source = inputs.x3
                x1_4_source = inputs.x4
                x1_5_source = inputs.x5
                x1_6_source = inputs.x6
                x1_7_source = inputs.x7
                x1_8_source = inputs.x8



                x9_target, x9_target_lm = inputs.x9, inputs.x9_lm

                # train the discriminator
                d_loss, d_losses = compute_d_loss_multi(
                    nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, masks=None)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

                # train the generator
                g_loss, g_losses = compute_g_loss_multi(
                    nets, args,x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, criterion_id, masks=None,
                    )
                self._reset_grad()
                g_loss.backward()

                optims.generator.step()
                optims.style_encoder.step()


            else:
                x1_source, x1_source_lm = inputs.x1, inputs.x_lm
                x2_target, x2_target_lm = inputs.x2, inputs.x2_lm
                lm = inputs.lm
                if args.masks:
                    masks = x2_target_lm
                else:
                    masks = None

                # train the discriminator
                if args.multi_discriminator:
                    d_loss, d_losses = compute_d_loss(
                        nets, args, x1_source, x2_target, x2_target_lm, lm, masks=masks)
                    self._reset_grad()
                    d_loss.backward()
                    optims.discriminator.step()
                    optims.discriminator2.step()

                # if args.landmark_loss:
                #     d_loss, d_losses = compute_d_loss(
                #         nets, args, x1_source, x2_target, x2_target_lm, masks=masks)
                #     self._reset_grad()
                #     d_loss.backward()
                #     optims.discriminator.step()
                #     optims.discriminator2.step()

                else:
                    d_loss, d_losses = compute_d_loss(
                        nets, args, x1_source, x2_target, x2_target_lm, lm, masks=masks)
                    self._reset_grad()
                    d_loss.backward()
                    optims.discriminator.step()



                # train the generator
                g_loss, g_losses = compute_g_loss(
                    nets, args, x1_source, x2_target, x2_target_lm, lm, criterion_id, masks=masks, facemodel = facemodel)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.style_encoder.step()




            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)

            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)



            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses, g_losses],
                                        ['D/', 'G/']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value

                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                for key, value in all_losses.items():
                    self.writer.add_scalar(key, value, i+1)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)

                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')

        self.writer.close()

    @torch.no_grad()
    def sample(self, loaders):


        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, args.latent_dim, 'test',args.multi))
        fname = ospj(args.result_dir, 'reconstruct.jpg')
        print('Working on {}...'.format(fname))

        if args.multi:
            utils.translate_and_reconstruct_multi(nets_ema, args, src.x1, src.x2, src.x3, src.x4, src.x5, src.x6, src.x7, src.x8 ,src.x9, src.x9_lm, fname)
        else:
            #utils.translate_and_reconstruct(nets_ema, args, src.x1, src.x2, src.x2_lm, fname)
            utils.translate_and_reconstruct(nets_ema, args, src.x1, src.x2, src.x2_lm, src.lm, fname)



    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')



def compute_d_loss(nets, args, x1_source, x2_target, x2_target_lm, lm, masks=None):

    if args.multi_discriminator:


        x2_target.requires_grad_()
        x2_target_lm.requires_grad_()


        _, real_out_1 = nets.discriminator(x2_target, x2_target_lm, lm)
        _, real_out_2 = nets.discriminator2(x2_target, x2_target_lm, lm)

        real_out = real_out_1 + real_out_2
        loss_real = adv_loss(real_out, 1)
        loss_reg = r1_reg(real_out, x2_target)

        # with fake images
        with torch.no_grad():
            s_trg = nets.style_encoder(x1_source)
            x_fake = nets.generator(x2_target_lm, lm, s_trg, masks=masks)


        _, fake_out_1 = nets.discriminator(x_fake, x2_target_lm, lm)
        _, fake_out_2 = nets.discriminator2(x_fake, x2_target_lm, lm)

        fake_out = fake_out_1 + fake_out_2
        loss_fake = adv_loss(fake_out, 0)

        loss = loss_real + loss_fake + args.lambda_reg * loss_reg

        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())
    # if args.landmark_loss:
    #     row=[]
    #     x2_target.requires_grad_()
    #     x2_target_lm.requires_grad_()


    #     s_trg = nets.style_encoder(x1_source)
    #     x_fake = nets.generator(x2_target_lm, s_trg, masks=masks)
    #     #img_fake =tensor_to_np(x_fake[0].unsqueeze(0))
    #     #img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
    #     #cv2.imwrite("./lm/test1.jpg",img_fake)
    #     batch_fake_lm = np.zeros((4,3,256,256))
    #     batch_fake_lm = torch.from_numpy(batch_fake_lm)
    #     batch_fake_lm = batch_fake_lm.float()
    #     for i in range(len(x_fake)):
    #         img = np.array((x_fake[i].cpu().detach().numpy().squeeze().transpose(1,2,0))*256, dtype=np.uint8)
    #         #img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            
    #         try:

    #             preds = fa.get_landmarks(img)
    #             # img = np.array((x_fake[0].cpu().detach().numpy().squeeze().transpose(1,2,0) + 1)*128, dtype=np.uint8)
    #             # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #             for idx in range(len(preds)):
    #                 lmrks = preds[idx]
    #                 for pts in range(lmrks.shape[0]):
    #                     pts_x = lmrks[pts, 0]
    #                     pts_y = lmrks[pts, 1]
    #                     row += [pts_x, pts_y]
    #             fake_lm = draw_lmmap(row)
    #             #cv2.imwrite("./lm/{}.jpg".format(i),fake_lm)
    #             fake_lm = toTensor(fake_lm)
    #             batch_fake_lm[i, :, :, :] = fake_lm
    #             #row = np.array(row)
    #         except:
    #             real_out_1 = nets.discriminator(x2_target, x2_target_lm)
    #             real_out = real_out_1
    #             loss_real = adv_loss(real_out, 1)
    #             loss_reg = r1_reg(real_out, x2_target)
    #             fake_out_1 = nets.discriminator(x_fake, x2_target_lm)
    #             fake_out = fake_out_1
    #             loss_fake = adv_loss(fake_out, 0)
    #             loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    #             return loss, Munch(real=loss_real.item(),
    #                             fake=loss_fake.item(),
    #                             reg=loss_reg.item())

    #     real_out_1 = nets.discriminator(x2_target, x2_target_lm)

    #     real_out = real_out_1
    #     loss_real = adv_loss(real_out, 1)
    #     loss_reg = r1_reg(real_out, x2_target)

    #     # with fake images



    #     fake_out_1 = nets.discriminator(x_fake, x2_target_lm)
    #     fake_out_2 = nets.discriminator2(batch_fake_lm, x2_target_lm)

    #     fake_out = fake_out_1 + fake_out_2
    #     loss_fake = adv_loss(fake_out, 0)

    #     loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    #     return loss, Munch(real=loss_real.item(),
    #                        fake=loss_fake.item(),
    #                        reg=loss_reg.item())

    else:
        # with real images
        x2_target.requires_grad_()
        out = nets.discriminator(x2_target, x2_target_lm, lm)
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, x2_target)

        # with fake images
        with torch.no_grad():
            if args.id_embed:
                s_fea = embedder(x1_source)
                s_trg = nets.mlp(s_fea)
            else:
                s_trg = nets.style_encoder(x1_source)
            x_fake = nets.generator(x2_target_lm, lm, s_trg, masks=masks)
        out = nets.discriminator(x_fake, x2_target_lm, lm)
        loss_fake = adv_loss(out, 0)

        loss = loss_real + loss_fake + args.lambda_reg * loss_reg

        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())


def compute_g_loss(nets, args, x1_source, x2_target, x2_target_lm, lm, criterion_id, masks=None, facemodel = None):

    # adversarial loss
    if args.finetune:
        with torch.no_grad():
            s_trg = nets.style_encoder(x1_source)

    else:
        s_trg = nets.style_encoder(x1_source)

    if args.multi_discriminator:


        x_fake = nets.generator(x2_target_lm, lm, s_trg, masks=masks)
        fake_fea_1, fake_out_1 = nets.discriminator(x_fake, x2_target_lm, lm)
        fake_fea_2, fake_out_2 = nets.discriminator2(x_fake, x2_target_lm, lm)

        out = fake_out_1 + fake_out_2
        loss_adv = adv_loss(out, 1)

        if args.pix2pix:

            real_fea_1, real_out_1 = nets.discriminator(x2_target, x2_target_lm)
            real_fea_2, real_out_2 = nets.discriminator2(x2_target, x2_target_lm)

            for num in range(6):
                if num == 0:
                    loss_fm_1 =  torch.mean(torch.abs(fake_fea_1[num] - real_fea_1[num]))
                    loss_fm_2 =  torch.mean(torch.abs(fake_fea_2[num] - real_fea_2[num]))
                else:
                    loss_fm_1 +=  torch.mean(torch.abs(fake_fea_1[num] - real_fea_1[num]))
                    loss_fm_2 +=  torch.mean(torch.abs(fake_fea_2[num] - real_fea_2[num]))
            loss_fm = loss_fm_1 + loss_fm_2
        else:
            loss_fm = 0

    else:
        x_fake = nets.generator(x2_target_lm, lm, s_trg, masks=masks)
        out = nets.discriminator(x_fake, x2_target_lm, lm)
        loss_adv = adv_loss(out, 1)

    # Pixel-wise L1 Loss
    loss_pixel_1 = torch.mean(torch.abs(x_fake - x2_target))

    ### VGG 19 recon loss
    if args.vgg19_recon_loss:
        vgg19_loss = torch.mean(vgg19(x_fake, x2_target))
    #l2 loss
    loss = nn.MSELoss()
    loss_l2 = loss(x_fake,x2_target)
    # ID Loss (vggface)
    loss_id = criterion_id(x_fake, x2_target)

    # ID Loss (vggface2)

    if args.vggface2:

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        facemodel.eval()
        fea_real, _ = facemodel(nn.functional.interpolate(x_fake, size=(224, 224), mode='bilinear')*255)
        fea_fake, _ = facemodel(nn.functional.interpolate(x2_target, size=(224, 224), mode='bilinear')*255)
        loss_id_2 = torch.mean(1-cos(fea_real,fea_fake))
    else:
        loss_id_2 = 0
    if args.landmark_loss:
        fake_row=[]
        s_trg = nets.style_encoder(x1_source)
        x_fake = nets.generator(x2_target_lm, lm, s_trg, masks=masks)
        #img_fake =tensor_to_np(x_fake[0].unsqueeze(0))
        #img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("./lm/test1.jpg",img_fake)
        batch_fake_lm = np.zeros((4,3,256,256))
        batch_fake_lm = torch.from_numpy(batch_fake_lm)
        batch_fake_lm = batch_fake_lm.float()
        for i in range(len(x_fake)):
            img = np.array((x_fake[i].cpu().detach().numpy().squeeze().transpose(1,2,0))*256, dtype=np.uint8)
            #img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            
            try:
                preds = fa.get_landmarks(img)
                # img = np.array((x_fake[0].cpu().detach().numpy().squeeze().transpose(1,2,0) + 1)*128, dtype=np.uint8)
                # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                for idx in range(len(preds)):
                    lmrks = preds[idx]
                    for pts in range(lmrks.shape[0]):
                        pts_x = lmrks[pts, 0]
                        pts_y = lmrks[pts, 1]
                        fake_row += [pts_x, pts_y]
            except:
                for x in range(68):
                    fake_row += [0, 0]
        real_row=[]
        batch_fake_lm = np.zeros((4,3,256,256))
        batch_fake_lm = torch.from_numpy(batch_fake_lm)
        batch_fake_lm = batch_fake_lm.float()
        for i in range(len(x2_target)):
            img = np.array((x2_target[i].cpu().detach().numpy().squeeze().transpose(1,2,0))*256, dtype=np.uint8)
            #img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            
            try:
                preds = fa.get_landmarks(img)
                # img = np.array((x_fake[0].cpu().detach().numpy().squeeze().transpose(1,2,0) + 1)*128, dtype=np.uint8)
                # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                for idx in range(len(preds)):
                    lmrks = preds[idx]
                    for pts in range(lmrks.shape[0]):
                        pts_x = lmrks[pts, 0]
                        pts_y = lmrks[pts, 1]
                        real_row += [pts_x, pts_y]
            except:
                for x in range(68):
                    real_row += [0, 0]
        loss_lm = np.zeros([])
        loss_lm =torch.from_numpy(loss_lm)
        loss_lm = loss_lm.type(torch.cuda.FloatTensor)
        for i in range(272):
	        loss_lm += torch.mean(torch.abs(torch.tensor(((fake_row[i*2]-real_row[i*2])**2+(fake_row[i*2+1]-real_row[i*2+1])**2)**1/2)))
        #land_loss = fake_row[]for i in range(68)
    
    #Style Consistency Loss
        
    if args.style_cyc:
        # Feature Matching Loss
        if args.pix2pix:
            s_trg_2 = nets.style_encoder(x_fake)


            loss_id_cyc = torch.mean(torch.abs(s_trg_2 - s_trg))
            loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * (loss_id + loss_id_2) + args.lambda_style_cyc * loss_id_cyc + args.lambda_fm *loss_fm 
            return loss, Munch(adv=loss_adv.item(),
                            pixel_1=loss_pixel_1.item(),
                            id=loss_id.item(), id_cyc=loss_id_cyc.item(),fm=loss_fm.item())
        else:
            if args.finetune:
                with torch.no_grad():
                    s_trg_2 = nets.style_encoder(x_fake)
                    loss_id_cyc = torch.mean(torch.abs(s_trg_2 - s_trg))
            else:
                s_trg_2 = nets.style_encoder(x_fake)
                loss_id_cyc = torch.mean(torch.abs(s_trg_2 - s_trg))

            loss = loss_adv + args.lambda_id * loss_id_2 + args.lambda_style_cyc * loss_id_cyc + 1 * loss_l2 + 10 * vgg19_loss
            #loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id *  loss_id_2 + args.lambda_style_cyc * loss_id_cyc + loss_lm
            # return loss, Munch(adv=loss_adv.item(),
            #                 pixel_1=loss_pixel_1.item(),lm_loss=loss_lm.item(),
            #                 id=loss_id_2.item(), id_cyc=loss_id_cyc.item())
            return loss, Munch(adv=loss_adv.item(),
                            l2_loss=loss_l2.item(),
                            id=loss_id_2.item(), id_cyc=loss_id_cyc.item(), vgg19_loss=vgg19_loss.item())
    elif args.pix2pix:

        loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id + args.lambda_fm *loss_fm
        return loss, Munch(adv=loss_adv.item(),
                        pixel_1=loss_pixel_1.item(),
                        id=loss_id.item(),fm=loss_fm.item())
    else:
        loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id + loss_pixel_lm
        return loss, Munch(adv=loss_adv.item(),
                        pixel_1=loss_pixel_1.item(),
                        id=loss_id.item())


def compute_d_loss_multi(nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, masks=None):

    # with real images
    x9_target.requires_grad_()
    out = nets.discriminator(x9_target, x9_target_lm)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x9_target)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x1_1_source)
        s_trg += nets.style_encoder(x1_2_source)
        s_trg += nets.style_encoder(x1_3_source)
        s_trg += nets.style_encoder(x1_4_source)
        s_trg += nets.style_encoder(x1_5_source)
        s_trg += nets.style_encoder(x1_6_source)
        s_trg += nets.style_encoder(x1_7_source)
        s_trg += nets.style_encoder(x1_8_source)
        s_trg_mean = s_trg/8



        x_fake = nets.generator(x9_target_lm, s_trg_mean, masks=masks)


    out = nets.discriminator(x_fake, x9_target_lm)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss_multi(nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm,criterion_id, masks=None):

    # adversarial loss
    s_trg = nets.style_encoder(x1_1_source)
    s_trg += nets.style_encoder(x1_2_source)
    s_trg += nets.style_encoder(x1_3_source)
    s_trg += nets.style_encoder(x1_4_source)
    s_trg += nets.style_encoder(x1_5_source)
    s_trg += nets.style_encoder(x1_6_source)
    s_trg += nets.style_encoder(x1_7_source)
    s_trg += nets.style_encoder(x1_8_source)

    s_trg_mean = s_trg/8

    x_fake = nets.generator(x9_target_lm, s_trg_mean, masks=masks)
    out = nets.discriminator(x_fake, x9_target_lm)
    loss_adv = adv_loss(out, 1)

    loss_pixel_1 = torch.mean(torch.abs(x_fake - x9_target))

    loss_id = criterion_id(x_fake, x9_target)

    loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id

    return loss, Munch(adv=loss_adv.item(),
                       pixel_1=loss_pixel_1.item(),
                       id=loss_id.item())





def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def draw_lmmap(landmark):


    img_3 = np.zeros((256, 256, 3))
    line_color = (255, 255, 255)
    line_width = 2
    lm_x = []
    lm_y = []
    for num in range(68):
        lm_x.append(landmark[num*2])
        if num == 0:
            pass
        lm_y.append(landmark[num*2+1])

    diff_1 = int(float(lm_x[42])) - int(float(lm_x[36]))
    diff_2 = int(float(lm_x[45])) - int(float(lm_x[39]))

    if diff_1<=30 and int(float(lm_x[30])) > int(float(lm_x[42])):

        for n in range(0, 12):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        # for n in range(22, 26):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(31, 33):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
                 (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
        # for n in range(42, 47):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        # cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
        #          (int(float(lm_x[47])), int(float(lm_y[47]))), line_color, line_width)
        for n in range(48, 51):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[51])), int(float(lm_y[51]))),
                 (int(float(lm_x[62])), int(float(lm_y[62]))), line_color, line_width)
        for n in range(60, 62):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
                 (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[66])), int(float(lm_y[66]))),
                 (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[57])), int(float(lm_y[57]))),
                 (int(float(lm_x[66])), int(float(lm_y[66]))), line_color, line_width)

        for n in range(57, 59):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
                 (int(float(lm_x[59])), int(float(lm_y[59]))), line_color, line_width)

    elif diff_2 <= 30 and int(float(lm_x[30])) < int(float(lm_x[42])):

        for n in range(4, 16):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        # for n in range(17, 21):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(33, 35):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        # for n in range(36, 41):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        # cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
        #          (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
                 (int(float(lm_x[47])), int(float(lm_y[47]))), line_color, line_width)
        for n in range(51, 57):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[51])), int(float(lm_y[51]))),
                 (int(float(lm_x[62])), int(float(lm_y[62]))), line_color, line_width)
        for n in range(62, 66):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                     (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[66])), int(float(lm_y[66]))),
                 (int(float(lm_x[57])), int(float(lm_y[57]))), line_color, line_width)



    else:


        for n in range(0, 16):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))),line_color, line_width)
        for n in range(31, 35):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
                    (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
                    (int(float(lm_x[47])), int(float(lm_y[47]))),line_color, line_width)
        for n in range(48, 59):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
                    (int(float(lm_x[59])), int(float(lm_y[59]))), line_color, line_width)
        for n in range(60, 67):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
                    (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)


    return img_3

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img=img
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

