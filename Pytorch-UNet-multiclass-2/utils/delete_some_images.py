from PIL import Image
from os.path import splitext
from os import listdir
from glob import glob
from tqdm import tqdm
import os
import numpy as np
imgs_dir = './data/imgs/'
masks_dir = './data/masks/'
ids = [splitext(file)[0] for file in listdir(imgs_dir)
            if not file.startswith('.')]

rm_files = 0
for id in tqdm(ids):
    idx = id
    mask_file = glob(masks_dir + idx + '.png')
    img_file = glob(imgs_dir + idx + '.png')

    mask = Image.open(mask_file[0])

    if np.unique(np.array(mask)).shape[0] == 1:
        os.remove(img_file[0])
        mask.close()
        os.remove(mask_file[0])
        rm_files +=1

print('DONE! Removed files: ',rm_files)
