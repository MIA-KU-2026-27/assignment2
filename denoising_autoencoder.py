from multiprocessing.spawn import freeze_support

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time

# this brings in the ChestMNIST dataset, so we can access training, validation and test images in pytorch dataloaders
from medmnist.dataset import ChestMNIST






# define the NN architecture
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # TODO

    def forward(self, x):
        #TODO

        return x




if __name__ == '__main__':
    # prevent potential bug when training
    freeze_support()

    # lower the number of epochs during development
    NUM_EPOCHS = 30
    # batch size = number of simultaneously used images during mini-batch-based stochastic gradient descent
    BATCH_SIZE = 128
    # noise level as a variance, given that our input images are normalized to [0,1]
    # try halving for IMG_SIZE = 28, to reduce the noise effect on the smaller images!
    NOISE_LEVEL = 0.001
    # IMG_SIZE = 64 gives access to higher resolution images. 
    # ATTENTION: stay with IMG_SIZE = 28 for development, especially if you don't have a GPU!
    IMG_SIZE = 64
    VISUALIZE = True

    # makes sure images are converted to pytorch tensors
    data_transform = transforms.Compose([transforms.ToTensor()])

    # here we load the images. if they are not yet downloaded, images get downloaded and stored in 'data' folder
    training_dataset = ChestMNIST(split="train", download=True, root='./data', size=IMG_SIZE, transform=data_transform)
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # validation and testing datasets are analyzed in batches of 1 image (with these datasets, the model is applied, not trained!)
    validation_dataset = ChestMNIST(split="val", download=True, root='./data', size=IMG_SIZE, transform=data_transform)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    testing_dataset = ChestMNIST(split="test", download=True, root='./data', size=IMG_SIZE, transform=data_transform)
    testing_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(training_dataset)

    if VISUALIZE == True:
        # obtain one batch of training images
        dataiter = iter(training_dataloader)
        for i in range(0,3):
            batch_of_images, _ = next(dataiter)

            # apply noising to the whole batch of images
            noisy_images = batch_of_images + (NOISE_LEVEL ** 0.5) * torch.randn_like(batch_of_images)

            # get one orig image from the batch
            img_orig = np.squeeze(batch_of_images.numpy()[0])
            print(i, "img orig: min=", np.min(img_orig), "max=", np.max(img_orig), "mean=", np.mean(img_orig), "std=", np.std(img_orig))

            # get one noisy image from the batch
            img_noised = np.squeeze(noisy_images.numpy()[0])
            print(i, "img noised: min=", np.min(img_noised), "max=", np.max(img_noised), "mean=", np.mean(img_noised), "std=", np.std(img_noised))

            fig, axarr = plt.subplots(1, 2)

            axarr[0].imshow(np.squeeze(img_orig), cmap='gray')
            axarr[0].set_title('Original Image')
            axarr[1].imshow(np.squeeze(img_noised), cmap='gray')
            axarr[1].set_title('Noisy Image')
            
            plt.show()


