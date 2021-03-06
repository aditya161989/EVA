from albumentations import *
from albumentations.augmentations.transforms import *
from albumentations.pytorch import ToTensor
import numpy as np

class TrainTransform():
  def __init__(self):
    self.train_transform = Compose([
      HorizontalFlip(),
      Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      Cutout(num_holes=1 ,max_h_size=4, max_w_size=4),
      ToTensor()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img