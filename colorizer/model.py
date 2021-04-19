import os, png
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time

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
  def __init__(self, flag, dataDir='./imagewoof2-320/', data_range=(0,1000)):
    assert(flag in ['train', 'val'])
    print("load "+ flag+" dataset start")
    print("    from: %s" % dataDir)
    self.dataset = []
    trainDir = './imagewoof2-320/' + flag

    # this is for getting row data, but unsure if we need it here
    # jpeg = TurboJPEG()

    filecount = 0
    for foldername in os.listdir(trainDir):
      folderpath = trainDir + "/" + foldername
      for filename in os.listdir(folderpath):
        filepath = folderpath + "/" + filename
        # resize the image to be 128 by 128 for now
        # forget about resizing back for now
        img_og = T.resize(Image.open(filepath), (128, 128))

        # discard grayscale images in dataset
        if (len(np.asarray(img_og).shape) == 2):
          continue
        # convert colored image to grayscale
        img_gray = T.Grayscale(img_og)

        # item in data set the first thing in the tuple is the grayscale image
        # the second part in the image is the ground truth, which would just be the original image
        self.dataset.append((img_gray, img_og))

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
    # if not self.onehot:
    #   label = torch.argmax(label, dim=0)
    # else:
    #   label = label.long()

    

    return torch.FloatTensor(img), torch.FloatTensor(label)



# -------------------------------- NEW BLOCK --------------------------------- #

N_CLASS=5
##############################################################################
# TODO: Change data_range so that train_data and val_data splits the 906     #
# samples under "train" folder. You can decide how to split.                 #
#                                                                            # 
# TODO: Adjust batch_size for loaders                                        #
##############################################################################
train_data = DogeDataset(flag='train', data_range=(0,800))
train_loader = DataLoader(train_data, batch_size=4)
val_data = DogeDataset(flag='train', data_range=(800,906))
val_loader = DataLoader(val_data, batch_size=4)
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
test_data = DogeDataset(flag='val', data_range=(0,114))
test_loader = DataLoader(test_data, batch_size=1)

# ap_loader for calculating Average Precision
ap_data = DogeDataset(flag='val', data_range=(0,114))
ap_loader = DataLoader(ap_data, batch_size=1)



# -------------------------------- NEW BLOCK --------------------------------- #

class Colorizer(nn.Module):
  def __init__(self, input_size=128):
    super(Colorizer, self).__init__()
    # TODO: implement
    # TODO: Implement upsampling layers
    self.OUTPUT_DIM = 3
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
        # output dimension is just the number of color channels,
        nn.Conv2d(128, self.OUTPUT_DIM, kernel_size=3, stride=1, padding=1),
        nn.Upsample(scale_factor=2) # Upwnsample x2
    )


  def forward(self, input):
    # Pass input through ResNet-gray to extract features
    return self.layers(input)


def train(trainloader, net, criterion, optimizer, device, epoch):
  '''
  Function for training.
  '''
  start = time.time()
  running_loss = 0.0
  cnt = 0
  net = net.train()
  for images, labels in tqdm(trainloader):
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    output = net(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    cnt += 1
  end = time.time()
  running_loss /= cnt
  print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
        (epoch, running_loss, end-start))
  return running_loss

def test(testloader, net, criterion, device):
  '''
  Function for testing.
  '''
  losses = 0.
  cnt = 0
  with torch.no_grad():
    net = net.eval()
    for images, labels in tqdm(testloader):
      images = images.to(device)
      labels = labels.to(device)
      output = net(images)
      loss = criterion(output, labels)
      losses += loss.item()
      cnt += 1
  print('\n',losses / cnt)
  return (losses/cnt)

def save_results(testloader, net, device, folder='output_rgb'):
  result = []
  cnt = 1
  os.makedirs(folder, exist_ok=True)
  with torch.no_grad():
    net = net.eval()
    cnt = 0
    for images, labels in tqdm(testloader):
      images = images.to(device)
      labels = labels.to(device)
      output = net(images)[0].cpu().numpy()
      c, h, w = output.shape
      assert(c == 3)# make sure there are 3 color channels

      # Save the predicted image
      plt.imsave('./{}/input{}.png'.format(folder, cnt), output)

      # Save the ground truth label image
      plt.imsave('./{}/predicted{}.png'.format(folder, cnt), labels[cnt])
      
      # y = np.argmax(output, 0).astype('uint8')
      # gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
      # save_label(y, './{}/y{}.png'.format(folder, cnt))
      # save_label(gt, './{}/gt{}.png'.format(folder, cnt))
      # plt.imsave('./{}/x{}.png'.format(folder, cnt),
      #            ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))
      cnt += 1

def plot_hist(trn_hist, val_hist):
  x = np.arange(len(trn_hist))
  plt.figure()
  plt.plot(x, trn_hist)
  plt.plot(x, val_hist)
  plt.legend(['Training', 'Validation'])
  plt.xticks(x)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

model = Colorizer()
criterion = nn.MSELoss() # We can play around with the loss function and optimizer type and write about this in our report
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0) # Play around with hyperparameters -- grid/random search?
num_epoch = 15
net = Colorizer(input_size=128).to(device)

print('\nStart training')
trn_hist = []
val_hist = []
for epoch in range(num_epoch):
  print('-----------------Epoch = %d-----------------' % (epoch+1))
  trn_loss = train(train_loader, net, criterion, optimizer, device, epoch+1)
  print('Validation loss: ')
  val_loss = test(val_loader, net, criterion, device)
  trn_hist.append(trn_loss)
  val_hist.append(val_loss)

plot_hist(trn_hist, val_hist)

# run on the test data set and output results
save_results(test_loader, net, device)