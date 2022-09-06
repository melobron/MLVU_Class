from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from glob import glob
import cv2


class GOPRO_Large_train_sharp(Dataset):
    def __init__(self, img_size=256):
        super(GOPRO_Large_train_sharp, self).__init__()

        self.sharp_path = './datasets/GOPRO_Large/sharp_train/'
        self.sharp_files = sorted(glob(os.path.join(self.sharp_path, '*.png')))

        self.sharp_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size,img_size)),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        sharp_numpy = cv2.cvtColor(cv2.imread(self.sharp_files[index]), cv2.COLOR_BGR2RGB)
        sharp_torch = self.sharp_transform(sharp_numpy)
        return sharp_torch

    def __len__(self):
        return len(self.sharp_files)


# class GOPRO_Large_test_sharp(Dataset):
#     def __init__(self):
#         super(GOPRO_Large_test_sharp, self).__init__()

#         self.sharp_path = './datasets/GOPRO_Large/sharp_test/'
#         self.sharp_files = sorted(glob(os.path.join(self.sharp_path, '*.png')))

#         self.sharp_transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#     def __getitem__(self, index):
#         sharp_numpy = cv2.cvtColor(cv2.imread(self.sharp_files[index]), cv2.COLOR_BGR2RGB)
#         sharp_torch = self.sharp_transform(sharp_numpy)
#         return sharp_torch

#     def __len__(self):
#         return len(self.sharp_files)

#
#
# class GOPRO_Large_train_blur_128(Dataset):
#     def __init__(self):
#         super(GOPRO_Large_train_blur_128, self).__init__()
#
#         self.blur_path = './datasets/GOPRO_Large/ddpm_train_128/blur/'
#         self.blur_files = sorted(glob(os.path.join(self.blur_path, '*.png')))
#
#         self.blur_transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
#
#     def __getitem__(self, index):
#         blur_numpy = cv2.cvtColor(cv2.imread(self.blur_files[index]), cv2.COLOR_BGR2RGB)
#         blur_torch = self.blur_transform(blur_numpy)
#         return blur_torch
#
#     def __len__(self):
#         return len(self.blur_files)
#
#
# class GOPRO_Large_test_blur_128(Dataset):
#     def __init__(self):
#         super(GOPRO_Large_test_blur_128, self).__init__()
#
#         self.blur_path = './datasets/GOPRO_Large/ddpm_test_128/blur/'
#         self.blur_files = sorted(glob(os.path.join(self.blur_path, '*.png')))
#
#         self.blur_transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
#
#     def __getitem__(self, index):
#         blur_numpy = cv2.cvtColor(cv2.imread(self.blur_files[index]), cv2.COLOR_BGR2RGB)
#         blur_torch = self.blur_transform(blur_numpy)
#         return blur_torch
#
#     def __len__(self):
#         return len(self.blur_files)