from PIL import Image
from os.path import splitext
from os import listdir
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import json
import argparse

def count_occurences(data_dir,n_classes):
    masks_dir = os.path.join(data_dir,'masks/')
    ids = [splitext(file)[0] for file in listdir(masks_dir)
                if not file.startswith('.')]


    classes_occ = {classe: 0 for classe in range(n_classes)}
    frequencies = {classe: 0.0 for classe in range(n_classes)}

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
            frequencies[key] = float(value/total)
    return frequencies

def get_args():
    parser = argparse.ArgumentParser(description='Compute frequencies of labels in the masks directory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir','-d',dest='data_dir',
                        help='path to data directory, masks must be stored in masks/ subdirectory and .png files',default = 'data/')
    parser.add_argument('--n_classes','-nc',type=int,required = True,dest='n_classes',
                        help='number of classes')
    parser.add_argument('--save', '-s', action='store_false',
                        help='save frequencies in a json file',
                        default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    frequencies = count_occurences(args.data_dir,args.n_classes)
    if args.save:
        with open(os.path.join(args.data_dir,'frequencies.json'),'w') as freqFile:
            freqFile.write(json.dumps(frequencies))
