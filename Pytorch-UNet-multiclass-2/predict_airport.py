import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from predict import predict_img,get_output_filenames,get_args,mask_to_image
from unet import UNet
from tqdm import tqdm



if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)
    n_classes = args.n_classes
    net = UNet(n_channels=3, n_classes=n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    for k, fn in enumerate(in_files):
        img = Image.open(fn)
        mask = np.array(img)[:,:,0]
        blockSize = 512
        wBlocks = img.size[0]//blockSize
        hBlocks = img.size[1]//blockSize

        for i in tqdm(range(wBlocks+1)):
            for j in range(hBlocks+1):
                h = blockSize if j < hBlocks else img.size[1]%blockSize
                w = blockSize if i < wBlocks else img.size[0]%blockSize

                cropped_img = img.crop((i*blockSize,j*blockSize,i*blockSize + w,j*blockSize + h))

                pred_mask = predict_img(net=net,
                           full_img=cropped_img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

                mask[j*blockSize:j*blockSize + h,i*blockSize:i*blockSize + w] = np.argmax(pred_mask,axis=0)


        if not args.no_save:
            out_fn = out_files[k]
            result = mask_to_image(mask,n_classes)
            print(out_files[k])
            result.save(out_files[k])
