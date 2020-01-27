from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) 
                    if not file.startswith('.')]
        print(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img):
        w, h = pil_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        print(img_trans)
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = "/Users/salimbaddou/PIE/Pytorch-UNet/" + self.masks_dir + idx + '.png'

        img_file = "/Users/salimbaddou/PIE/Pytorch-UNet/" + self.imgs_dir + idx + '.png'


        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}



class Dataset2(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Dataset2(Dataset):

    def __init__(self,imgs_dir, masks_dir, scale):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample