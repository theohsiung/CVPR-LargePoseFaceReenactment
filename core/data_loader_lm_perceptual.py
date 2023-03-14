"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root[0])
        self.samples.sort()

        self.samples_2 = listdir(root[1])
        self.samples_2.sort()

        self.samples_3 = listdir(root[2])
        self.samples_3.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        fname_2 = self.samples_2[index]
        fname_3 = self.samples_3[index]


        img = Image.open(fname).convert('RGB')
        img_2 = Image.open(fname_2).convert('RGB')
        img_3 = Image.open(fname_3).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
        return img, img_2, img_3

    def __len__(self):
        return len(self.samples)

class LMDataset(data.Dataset):
    def __init__(self, root, transform=None, train_data = 'mpie',multi=False,test='train'):

        self.multi = multi
        self.test = test
        if multi:

            self.train_data = train_data
            self.transform = transform
            self.targets = None
            self.samples = []
            self.samples2 = []
            self.samples3 = []
            self.samples4 = []
            self.samples5 = []
            self.samples6 = []
            self.samples7 = []
            self.samples8 = []
            self.samples9 = []

            with open(root) as F:

                for line in F:
                    line = line.strip('\n')
                    # print(line.split(' '))
                    self.samples.append(line.split(' ')[0])
                    self.samples2.append(line.split(' ')[1])
                    self.samples3.append(line.split(' ')[2])
                    self.samples4.append(line.split(' ')[3])
                    self.samples5.append(line.split(' ')[4])
                    self.samples6.append(line.split(' ')[5])
                    self.samples7.append(line.split(' ')[6])
                    self.samples8.append(line.split(' ')[7])
                    self.samples9.append(line.split(' ')[8])

        else:

            self.train_data = train_data
            if self.train_data == 'rafd':

                if self.test == 'train':
                    self.transform = transform
                    self.targets = None
                    self.samples = []
                    self.samples2 = []
                    with open(root) as F:
                        for line in F:
                            line = line.strip('\n')
                            self.samples.append(line.split(' ')[0])
                            self.samples2.append(line.split(' ')[1])

                else:
                    self.transform = transform
                    self.targets = None
                    self.samples = []
                    self.samples2 = []
                    self.samples3 = []
                    with open(root) as F:
                        for line in F:
                            line = line.strip('\n')
                            self.samples.append(line.split(' ')[0])
                            self.samples2.append(line.split(' ')[1])
                            #self.samples3.append(line.split(' ')[2])

            else:
                self.transform = transform
                self.targets = None
                self.samples = []
                self.samples2 = []
                with open(root) as F:
                    n=0
                    try:
                        for line in F:
                            
                            line = line.strip('\n')
                            # print(line.split(' '))
                            self.samples.append(line.split(' ')[0])
                            self.samples2.append(line.split(' ')[1])
                    except Exception as e:
                        n+=1
                        print('error {} images'.format(n))
                        print(self.samples[-1])
                        print(self.samples2[-1])
                        del self.samples[-1]
                        del self.samples2[-1]
                        pass



    def __getitem__(self, index):
        if self.multi:
            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]
            fname3_folder = self.samples3[index]
            fname4_folder = self.samples4[index]
            fname5_folder = self.samples5[index]
            fname6_folder = self.samples6[index]
            fname7_folder = self.samples7[index]
            fname8_folder = self.samples8[index]
            fname9_folder = self.samples9[index]







            if self.train_data == 'vox1':


                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                fname4 = fname4_folder
                fname5 = fname5_folder
                fname6 = fname6_folder
                fname7 = fname7_folder
                fname8 = fname8_folder
                fname9 = fname9_folder


                fname_lm9 = fname9.split('unzippedFaces')[0] + 'lm/unzippedFaces' + fname9.split('unzippedFaces')[1]



            img = Image.open(fname).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            img3 = Image.open(fname3).convert('RGB')
            img4 = Image.open(fname4).convert('RGB')
            img5 = Image.open(fname5).convert('RGB')

            img6 = Image.open(fname6).convert('RGB')
            img7 = Image.open(fname7).convert('RGB')
            img8 = Image.open(fname8).convert('RGB')
            img9 = Image.open(fname9).convert('RGB')

            img_lm9 = Image.open(fname_lm9).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
                img4 = self.transform(img4)
                img5 = self.transform(img5)
                img6 = self.transform(img6)
                img7 = self.transform(img7)
                img8 = self.transform(img8)
                img9 = self.transform(img9)

                img_lm9 = self.transform(img_lm9)
                # img_lm = self.transform(img_lm)

            return img, img2, img3, img4,img5,img6,img7,img8, img9, img_lm9

        else:

            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]

            if self.train_data == 'mpie':


                fname = fname_folder
                fname2 = fname2_folder

                fname_lm = fname.split('mpie_s2_crop_256')[0] + 'mpie_s2_LM_256' +fname.split('mpie_s2_crop_256')[1]
                fname_lm2 = fname2.split('mpie_s2_crop_256')[0] + 'mpie_s2_LM_256' +fname2.split('mpie_s2_crop_256')[1]


            elif self.train_data == 'rafd':
                fname = fname_folder
                fname2 = fname2_folder
                fname_lm = fname.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname.split('rafd_crop_256')[1]
                fname_lm2 = fname2.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname2.split('rafd_crop_256')[1]

                # if self.test == 'train':
                #     fname = fname_folder
                #     fname2 = fname2_folder
                #     # fname_lm = fname.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname.split('rafd_crop_256')[1]
                #     # fname_lm2 = fname2.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname2.split('rafd_crop_256')[1]
                #     fname_lm = fname.split('rafd_crop_256')[0] + 'rafd_pncc_256' + fname.split('rafd_crop_256')[1]
                #     fname_lm2 = fname2.split('rafd_crop_256')[0] + 'rafd_pncc_256' + fname2.split('rafd_crop_256')[1]
                #     #print(fname_lm)
                # else:
                #     fname = fname2_folder
                #     fname2 = self.samples3[index]
                #     fname_lm = fname_folder
                #     fname_lm2 = fname_folder


            elif self.train_data == 'vox1':

                fname = fname_folder
                fname2 = fname2_folder
                
                fname = fname.replace('\\','/')
               
                fname2 = fname2.replace('\\','/')
                # lm = fname2.split('images')[0] + 'images_lm' +fname2.split('images')[1]
                # pncc = fname2.split('images')[0] + 'images_pncc' +fname2.split('images')[1]
                # if fname.split('/')[2] == 'MPIE':
                #     fname_lm = fname.split('mpie_s2_crop_256')[0] + 'mpie_s2_LM_256' + \
                #                 fname.split('mpie_s2_crop_256')[1]
                # if fname.split('/')[5] == 'vox1':
                #     n1 = (fname.split('voxceleb1_crop_256')[1]).split("/1.6")[0]
                #     v1 = (fname.split('voxceleb1_crop_256')[1]).split("/1.6")[1]
                #     fname_lm = fname.split('voxceleb1_crop_256')[0] + 'voxceleb1_pncc_256' +n1+v1
                # if fname.split('/')[2] == 'mpie':
                #     # fname_lm  = fname.split('mpie_s2_arcface_crop_256')[0] + 'mpie_s2_arcface_LM_256' + \
                #     #            fname.split('mpie_s2_arcface_crop_256')[1]
                #     fname_lm  = fname.split('MPIE_WarpAffine_250_250_Clean')[0] + 'mpie_wrap_pncc' + \
                #                fname.split('MPIE_WarpAffine_250_250_Clean')[1]
                #     # n = (fname.split('mpie_s2_arcface_crop_256')[0])
                #     # v = (fname.split('mpie_s2_arcface_crop_256')[1])
                #     # fname_lm = n + 'mpie_arcface_pncc'+v
                # if fname.split('/')[2] == 'Vox2_test':
                #     fname_lm = fname.split('Vox2_test')[0] + 'Vox2lm_test/LM_256' + fname.split('Vox2_test')[1]
                # if fname.split('/')[2] == 'Vox2_test(25)':
                #     fname_lm = fname.split('Vox2_test(25)')[0] + 'Vox2lm_test(25)final/LM_256' + fname.split('Vox2_test(25)')[1]
                # if fname.split('/')[2] == 'Vox2':
                #     fname_lm = fname.split('Vox2')[0] + 'Vox2lm/LM_256' + fname.split('Vox2')[1]

                # else:

                #     fname_lm = fname.split('unzippedFaces')[0] + 'voxceleb1_LM_width_2' + fname.split('unzippedFaces')[1]


                if fname2.split('/')[5] == 'vox1':
                    n2 = (fname2.split('voxceleb1_crop_256')[1]).split("/1.6")[0]
                    v2 = (fname2.split('voxceleb1_crop_256')[1]).split("/1.6")[1]
                    pncc = fname2.split('voxceleb1_crop_256')[0] + 'voxceleb1_pncc_256' +n2+v2
                    lm = fname2.split('voxceleb1_crop_256')[0] + 'voxceleb1_LM_256' + fname2.split('voxceleb1_crop_256')[1]
                if fname2.split('/')[5] == 'mpie_s2_arcface_256':
                    pncc = fname2.split('mpie_s2_arcface_crop_256')[0] + 'mpie_arcface_pncc' + \
                                fname2.split('mpie_s2_arcface_crop_256')[1]
                    lm = fname2.split('mpie_s2_arcface_crop_256')[0] + 'mpie_s2_arcface_LM_256' + \
                                fname2.split('mpie_s2_arcface_crop_256')[1]
                # if fname2.split('/')[2] == 'MPIE':
                #     fname_lm2 = fname2.split('mpie_s2_crop_256')[0] + 'mpie_s2_LM_256' + \
                #                 fname2.split('mpie_s2_crop_256')[1]
                # if fname2.split('/')[2] == 'mpie':
                #     # fname_lm2 = fname2.split('mpie_s2_arcface_crop_256')[0] + 'mpie_s2_arcface_LM_256' + \
                #     #            fname2.split('mpie_s2_arcface_crop_256')[1]
                #     fname_lm2 = fname2.split('MPIE_WarpAffine_250_250_Clean')[0] + 'mpie_wrap_pncc' + \
                #                fname2.split('MPIE_WarpAffine_250_250_Clean')[1]
                #     # n2 = (fname2.split('mpie_arcface_pncc')[0])
                #     # v2 = (fname2.split('mpie_arcface_pncc')[1])
                #     # fname_lm2 = n2 + 'mpie_arcface_pncc'+v2
                # if fname2.split('/')[2] == 'Vox2_test':
                #     fname_lm2 = fname2.split('Vox2_test')[0] + 'Vox2lm_test/LM_256'+ fname2.split('Vox2_test')[1]
                # if fname2.split('/')[2] == 'Vox2_test(25)':
                #     fname_lm2 = fname2.split('Vox2_test(25)')[0] + 'Vox2lm_test(25)final/LM_256' + fname2.split('Vox2_test(25)')[1]
                # if fname2.split('/')[2] == 'Vox2':
                #     fname_lm2 = fname2.split('Vox2')[0] + 'Vox2lm/LM_256' + fname2.split('Vox2')[1]
                # else:


                #     fname_lm2 = fname2.split('unzippedFaces')[0] + 'voxceleb1_LM_width_2' + \
                #                 fname2.split('unzippedFaces')[1]
            elif self.train_data == 'vox2':

                fname = fname_folder
                fname2 = fname2_folder

                fname = fname.replace('\\','/') #pncc

                fname2 = fname2.replace('\\','/') #pncc
                if fname2.split('/')[5] == 'mpie_s2_arcface_256':
                    pncc = fname2.split('mpie_s2_arcface_crop_256')[0] + 'mpie_arcface_pncc' + \
                                fname2.split('mpie_s2_arcface_crop_256')[1]
                    lm = fname2.split('mpie_s2_arcface_crop_256')[0] + 'mpie_s2_arcface_LM_256' + \
                                fname2.split('mpie_s2_arcface_crop_256')[1]

                #pncc = fname2.split('crop_256')[0] + 'pncc_256' + fname2.split('crop_256')[1]
                #lm = fname2.split('crop_256')[0] + 'LM_256' + fname2.split('crop_256')[1]

            # sor = fname.split(",")[0]
            # flow = fname.split(",")[1]
            # ref = fname2
            img = Image.open(fname).convert('RGB')
            img_lm = Image.open(pncc).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            img_lm2 = Image.open(pncc).convert('RGB')
            lm = Image.open(lm).convert('RGB')



            if self.transform is not None:


                img = self.transform(img)
                img2 = self.transform(img2)
                img_lm2 = self.transform(img_lm2)
                img_lm = self.transform(img_lm)
                lm = self.transform(lm)


            return img, img2, img_lm, img_lm2, lm



    def __len__(self):
        return len(self.samples)





def get_train_loader_vgg(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, shuffle=True, train_data = 'mpie', multi=False):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])

    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           pin_memory=True,
                           drop_last=True)



def get_eval_loader_vgg(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=False,
                    num_workers=4, drop_last=False, train_data = 'mpie',multi=False, mode='train'):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299

    else:
        height, width = img_size, img_size


    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor()

    ])



    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi,test=mode)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader_vgg(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, train_data='mpie',multi=False, mode='train'):
    print('Preparing DataLoader for the generation phase...')

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])


    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi,test=mode)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True)

def get_eval_loader_2(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=False,
                    num_workers=4, drop_last=False, train_data = 'mpie',multi=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    dataset = DefaultDataset(root, transform=transform)
    # dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


class InputFetcher:
    def __init__(self, loader, latent_dim=16, mode='', multi=False):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.multi = multi

    def _fetch_inputs(self):
        if self.multi:
            try:
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm = next(self.iter)
            return x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm
        else:
            try:
                x1, x2, x_lm, x2_lm, lm = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x1, x2, x_lm, x2_lm, lm= next(self.iter)
            return x1, x2, x_lm, x2_lm, lm



    def __next__(self):
        if self.multi:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm = self._fetch_inputs()

            inputs = Munch(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, x8=x8, x9=x9, x9_lm=x9_lm)

            return Munch({k: v.to(self.device)
                          for k, v in inputs.items()})
        else:
            x1, x2, x_lm, x2_lm, lm= self._fetch_inputs()

            inputs = Munch(x1=x1, x2=x2, x_lm=x_lm, x2_lm=x2_lm, lm=lm)

            return Munch({k: v.to(self.device)
                          for k, v in inputs.items()})
