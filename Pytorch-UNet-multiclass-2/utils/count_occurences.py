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

n_classes = 4
classes_occ = {classe: 0 for classe in range(n_classes)}

for id in tqdm(ids):
    idx = id
    mask_file = glob(masks_dir + idx + '.png')


    mask = Image.open(mask_file[0])

    uniqueValues, occurCount = np.unique(mask, return_counts=True)
    for val,occ in zip(uniqueValues,occurCount):
        classes_occ[val] += occ

total = sum([classes_occ[k] for k in classes_occ])
for key, value in classes_occ.items():
        print('Label {}: {:.3f}'.format(key,value/total))
