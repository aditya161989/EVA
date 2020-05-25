import torch.nn.functional as F
from dice_loss import dice_loss

def calc_loss(pred, target):

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)


    return dice