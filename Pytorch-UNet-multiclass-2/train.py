import argparse
import logging
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.frequencies import count_occurences
from torch.utils.data import DataLoader, random_split
data_dir = '/data/data'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=2,
              lr=0.01,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              data_dir = data_dir,
              config = None):

    dir_img = data_dir + 'imgs/'
    dir_mask = data_dir + 'masks/'
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    if config:
        with open(os.path.join(writer.log_dir,'config.json'),'w') as configFile:
            configFile.write(json.dumps(config.__dict__))

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if args.weighted:
        try:
            with open(os.path.join(data_dir,'frequencies.json'),'r') as freqFile:
                class_weights = torch.FloatTensor([1/f for _,f in json.load(freqFile).items()]).to(device)
        except FileNotFoundError:
                print('Computing adapted class weights...')
                frequencies = count_occurences(data_dir,net.n_classes)
                with open(os.path.join(data_dir,'frequencies.json'),'w') as freqFile:
                    freqFile.write(json.dumps(frequencies))
                class_weights = torch.FloatTensor([1/f for _,f in frequencies.items()]).to(device)



    else:
        class_weights = torch.FloatTensor([1]*net.n_classes)

    # if net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss(weight = class_weights)
    # else:
    #     criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights[1])
     criterion = nn.CrossEntropyLoss(weight = class_weights)


    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image'].type(dtype)
                true_masks = batch['mask'].type(dtype)
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)


                masks_pred = net(imgs)
                true_masks = true_masks.view(true_masks.shape[0],true_masks.shape[2],true_masks.shape[3])
                #
                #print("mask_pred: ",masks_pred.shape)
                #print("mask_true: ",true_masks.shape)
                #
                # print("mask_pred: ",masks_pred.dtype)
                # print("mask_true: ",true_masks.dtype)




                loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # print(global_step)
                # print(len(dataset))
                # print(batch_size)
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val, class_weights)
                    #if net.n_classes > 1:
                    logging.info('Validation cross entropy: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)

                    # else:
                    #     logging.info('Validation Dice Coeff: {}'.format(val_score))
                    #     writer.add_scalar('Dice/test', val_score, global_step)

                    #writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                    #     writer.add_images('masks/true', true_masks, global_step)
                    #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--weighted','-w',action='store_true',dest='weighted',
                        help='use of weighted cross entropy loss',default=False)
    parser.add_argument('--data_dir','-d',dest='data_dir',
                        help='path to data used for training',default = data_dir)
    parser.add_argument('--n_classes','-nc',type=int,required = True,dest='n_classes',
                        help='number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=args.n_classes)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val,
                  data_dir = args.data_dir,
                  config = args)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
