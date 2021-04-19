import os, png
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from PIL import Image
from tqdm import tqdm # Displays a progress bar

from torch import nn, optim
from torchsummary import summary
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset, Subset, DataLoader, random_split


# -------------------------------- NEW BLOCK --------------------------------- #
if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = torch.device('cuda:0')
else:
    raise Exception("WARNING: Could not find GPU! Using CPU only. \
To enable GPU, please to go Edit > Notebook Settings > Hardware \
Accelerator and select GPU.")


# -------------------------------- NEW BLOCK --------------------------------- #
class DogeDataset(Dataset):
    #TODO: refactor data range and n_class, also unsure what onehot is
  def __init__(self, flag, dataDir='./imagewoof2-320/', data_range=(0,1000), n_class=5, 
               onehot=False):
    self.onehot = onehot
    assert(flag in ['train', 'val'])
    print("load "+ flag+" dataset start")
    print("    from: %s" % dataDir)
    self.dataset = []
    trainDir = './imagewoof2-320/train'

    # this is for getting row data, but unsure if we need it here
    # jpeg = TurboJPEG()

    filecount = 0
    for foldername in os.listdir(trainDir):
      folderpath = trainDir + "/" + foldername
      for filename in os.listdir(folderpath):
        filepath = folderpath + "/" + filename
        img_og = Image.open(filepath)

        # discard grayscale images in dataset
        if (len(np.asarray(img_og).shape) == 2):
          continue
        # convert colored image to grayscale
        img_gray = T.Grayscale(img_og)
        img = np.asarray(img_gray).astype("f").transpose(2, 0, 1)/128.0-1.0 # normalize image

        # create labels --> png, normalize, append to dataset
        png_path = filepath[:-4] + ".png"
        img_og.save(png_path)
        
        pngreader = png.Reader(filename=png_path)
        w,h,row,info = pngreader.read()
        label = np.array(list(row)).astype('uint8')

        # TODO: normalize the input images to our color range
        # Normalize input image
        # Convert to n_class-dimensional onehot matrix
        label_ = np.asarray(label)
        label = np.zeros((n_class, img.shape[1], img.shape[2])).astype("i")
        for j in range(n_class):
            label[j, :] = label_ == j
        
        self.dataset.append((img, label))

        # this is rather jank, works for now, maybe refactor
        filecount += 1
        if filecount == data_range[1]:
          print("load dataset done")
          return

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    # from hw5 part 3
    img, label = self.dataset[index]
    label = torch.FloatTensor(label)
    if not self.onehot:
      label = torch.argmax(label, dim=0)
    else:
      label = label.long()

    # from discord, probs can discard this 

    # path, target = self.dataset[index]
    # img = self.loader(path)
    # if self.transform is not None:
    #   img_original = self.transform(img)
    #   img_original = np.asarray(img_original)
    #   img_lab = rgb2lab(img_original)
    #   img_lab = (img_lab + 128) / 255
    #   img_ab = img_lab[:, :, 1:3]
    #   img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
    #   img_original = rgb2gray(img_original)
    #   img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    # if self.target_transform is not None:
    #   target = self.target_transform(target)
    # return img_original, img_ab, target

    return torch.FloatTensor(img), torch.LongTensor(label)



# -------------------------------- NEW BLOCK --------------------------------- #

N_CLASS=5
##############################################################################
# TODO: Change data_range so that train_data and val_data splits the 906     #
# samples under "train" folder. You can decide how to split.                 #
#                                                                            # 
# TODO: Adjust batch_size for loaders                                        #
##############################################################################
train_data = DogeDataset(flag='train', data_range=(0,800), onehot=False)
train_loader = DataLoader(train_data, batch_size=4)
val_data = DogeDataset(flag='train', data_range=(800,906), onehot=False)
val_loader = DataLoader(val_data, batch_size=4)
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
test_data = DogeDataset(flag='val', data_range=(0,114), onehot=False)
test_loader = DataLoader(test_data, batch_size=1)

# ap_loader for calculating Average Precision
ap_data = DogeDataset(flag='val', data_range=(0,114), onehot=True)
ap_loader = DataLoader(ap_data, batch_size=1)



# -------------------------------- NEW BLOCK --------------------------------- #

class Colorizer(nn.Module):
  def __init__(self, input_size=128):
    super(Colorizer, self).__init__()
    # TODO: implement
    # TODO: Implement upsampling layers
    self.layers = nn.Sequential(
        # Downsampling layers (Downsampled x4)
        # 1 -> 128 -> 128 -> MaxPool2d
        nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(128, 128, kernel_size=3, stride=3, padding=1),
        nn.Relu(),
        nn.MaxPool2d(kernel_size=3, stride=2), # Downsample x2

        # 128 -> 256 -> 256 -> MaxPool2d
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.Relu(),
        nn.MaxPool2d(kernel_size=3, stride=2), # Downsample x2

        # 128 -> 256 -> 256 -> Batchnorm
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.BatchNorm2d(256),

        # 256 -> 512 -> 512 -> Batchnorm
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.BatchNorm2d(512),

        # 512 -> 512 -> 512 -> Batchnorm
        # Dialation is the space between kernel entries, (dialation=1 looks like a checkerboard)
        nn.Conv2d(512, 512, kernel_size=3, dialation=1, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(512, 512, kernel_size=3, dialation=1, stride=1, padding=1),
        nn.Relu(),
        nn.BatchNorm2d(512),

        # 512 -> 512 -> 512 -> Batchnorm
        nn.Conv2d(512, 512, kernel_size=3, dialation=2, stride=1, padding=2),
        nn.Relu(),
        nn.Conv2d(512, 512, kernel_size=3, dialation=2, stride=1, padding=2),
        nn.Relu(),
        nn.BatchNorm2d(512),

        #Upsampling layers
        # 512 -> 512 -> 512 -> Batchnorm
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.BatchNorm2d(512),
        nn.Upsample(scale_factor=2), # Upwnsample x2
        
        # 512 -> 512 -> 512 -> Batchnorm -> Upsample
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.BatchNorm2d(512),
        nn.Upsample(scale_factor=2), # Upwnsample x2

        # 512 -> 512 -> 512 -> Batchnorm
        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.Relu(),
        nn.BatchNorm2d(128),
        
        # TODO: What is output dimension?
        nn.Conv2d(128, OUTPUT_DIM, kernel_size=3, stride=1, padding=1),
        nn.Upsample(scale_factor=2) # Upwnsample x2
    )


  def forward(self, input):
    # Pass input through ResNet-gray to extract features
    self.layers(input)

model = Colorizer()
criterion = nn.MSELoss() # We can play around with the loss function and optimizer type and write about this in our report
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0) # Play around with hyperparameters -- grid/random search?
num_epoch = 10 

def save_label(label, path):
  '''
  Convert label into actual pixel color
  '''
  '''colormap = [
    '#000000',
    '#FFFFFF'
  ]
  colors = colorlabels[label]# this should be LAB color or RGB?
  w = png.Writer(label.shape[1], label.shape[0], palette=colormap)
  with open(path, 'wb') as f:
    w.write(f, label)
  '''