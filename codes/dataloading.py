import os
import torch
from torchvision import datasets
from torchvision import transforms
import random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torch.utils.data.sampler import SubsetRandomSampler
def mod_crop(image, scale):
    image = np.array(image)
    h, w = image.shape[0], image.shape[1]
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    return Image.fromarray(image[:h, :w])
class ImageIODataset(torch.utils.data.Dataset):

    def __init__(self, label_path,valid_flag,bicubic_flag, scale,label_transform=None):

        self.label_path = label_path
        self.total_label_imgs = os.listdir(label_path) # Get all img directories
        self.tensor_transform = transforms.ToTensor()
        self.center_crop=transforms.CenterCrop(size=(48 ,48))
        self.valid_flag=valid_flag
        self.label_transform=label_transform
        self.scale=scale
        self.bicubic_flag=bicubic_flag

    def __len__(self):
        return len(self.total_label_imgs)


    def __getitem__(self, idx):
        label_loc = os.path.join(self.label_path, self.total_label_imgs[idx])

        img = (Image.open(label_loc))#.convert("YCbCr") # Open image as RGB
        #if self.valid_flag==False:
        #  img = self.center_crop(img)
        # if self.valid_flag:
        img = mod_crop(img,self.scale)
        if random.random() > 0.5 and self.valid_flag==False:
          img = transforms.functional_pil.vflip(img)
        if random.random() > 0.5 and self.valid_flag==False:
          img = transforms.functional_pil.hflip(img)
        w, h = img.size
        input_img = img.resize((w//self.scale, h//self.scale),resample=Image.BICUBIC)
        if self.bicubic_flag:
          input_img = input_img.resize((w, h), resample=Image.BICUBIC)

        label_img = self.tensor_transform(img)
        input_img = self.tensor_transform(input_img)
        return input_img, label_img

def get_train_valid_loader(batch_size, random_seed, shuffle=True, num_workers=2, pin_memory=False, bicubic_flag=False, scale=3):
  """
  Gets training and validating loaders
  """

  # Loading datasets
  train_dataset = ImageIODataset(label_path="./data/train/train_sliced/",valid_flag=False,bicubic_flag=bicubic_flag,scale=scale)
  valid_dataset = ImageIODataset(label_path="./data/valid/valid_full_dimensions/",valid_flag=True,bicubic_flag=bicubic_flag,scale=scale)
  # Setting the indices
  num_train = len(train_dataset)
  indices = list(range(num_train))
  train_portion = int(np.floor(num_train))

  # Randomly shuffling the indices
  if shuffle:
      np.random.seed(random_seed)
      np.random.shuffle(indices)
  # Sampling the elements randomly from the indices
  train_idx = indices[:train_portion]
  train_sampler = SubsetRandomSampler(train_idx)

  # Iterate over the dataset by batches
  train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)
  valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1,num_workers=num_workers, pin_memory=pin_memory)

  # How many samples are in each split
  print("\nNumber of training samples: {:d}\nNumber of validation samples: {:d}\n".format(len(train_idx),len(valid_loader)))
  return train_loader, valid_loader
