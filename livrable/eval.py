import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device, n_val, class_weights):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image'].type(dtype)
            true_masks = batch['mask'].type(dtype)

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            true_masks = true_masks.view(true_masks.shape[0],true_masks.shape[2],true_masks.shape[3])
            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                #if net.n_classes > 1:
                tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0),weight = class_weights).item()

                # else:
                #     pred = (pred > 0.5).float()
                #     tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return tot / n_val
