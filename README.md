[[python 3.7]](https://www.python.org/downloads/release/python-370/),[[cuda 11.3]](https://pytorch.org/get-started/previous-versions/)
# PASL-for-Large-Pose-Face-Reenactment
Abstract: We propose the Pose Adapted Shape Learning (PASL)
for large-pose face reenactment. Different from previous
approaches that consider one single face encoder for iden-
tity preservation during training, we propose multiple Pose-
Adapted face Encodes (PAEs) for improving identity preser-
vation across large pose. The PASL model consists of three
modules, namely the Pose-Adapted face Encoder (PAE), the
Disentangled Shape Encoder (DSE) and the Target Face
Generator (TFG). Given a source face and a reference face,
the DSE generates a fused shape code that combines the
source identity and reference action in the shape space. The
TFG takes the fused shape code and source image as inputs
to generate the desired reenacted face by minimizing a par-
ticular set of loss functions. The core part of the loss func-
tions is the PAE-based identity loss, which appropriately
addresses the pose imbalance issue in the face datasets used
to train general face encoders. To better demonstrate the
large-pose performance, we propose the MPIE-LP (Large
Pose) and VoxCeleb2-LP subsets made from the original
MPIE and VoxCeleb2 datasets. Experiments show that the
proposed approach can better handle large-pose face reen-
actment with performance better than state of the art.
![mpie_git](https://user-images.githubusercontent.com/127723538/224975580-e78dd2f1-edd8-45c1-bb2a-69f86e9ad39e.png)
![demo_show](https://user-images.githubusercontent.com/127723538/224976806-12ca10aa-1c1e-4cb0-9804-c931e5c1bc7f.gif)
# Demo
In this demonstration, you will serve as the reference face, allowing you to manipulate the source face and make it mimic your pose.
- To install the dependencies, please follow these steps:
`pip install -r requirements.txt` 
- Due to the large size of our pretrained model, it cannot be stored on GitHub. Please download the model from website instead.
https://drive.google.com/drive/folders/1ZDn0LVMjwm5FJXnZscSSHOCOvJMtI9JW?usp=sharing
- After downloading, unzip the file and move it to the main directory `./checkpoints` 
- To experience the live demo, please prepare a USB camera.
- Execute the following command `python demo.py`
- To optimize performance, we highly recommend using the same version as ours.
