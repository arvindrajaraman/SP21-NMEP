import numpy as np
import PIL
from PIL import Image
import glob
import torch
from torch.utils.data.dataset import Dataset

'''
Pytorch uses datasets and has a very handy way of creatig dataloaders in your main.py
Make sure you read enough documentation.
'''
class Data(Dataset):
    def __init__(self, data_dir):
         
        #gets the data from the directory
        self.image_list = glob.glob(data_dir + '*')
		#calculates the length of image_list
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        image = Image.open(single_image_path)
        # image = self.image_list[index]
        images = np.vstack([np.asarray(image.rotate(0)).transpose(2, 0, 1) / 255,
                            np.asarray(image.rotate(90)).transpose(2, 0, 1) / 255,
                            np.asarray(image.rotate(180)).transpose(2, 0, 1) / 255,
                            np.asarray(image.rotate(270)).transpose(2, 0, 1) / 255])
        '''
		# TODO: Convert your numpy to a tensor and get the labels
		'''
        image_tensor = torch.from_numpy(images).float()
        
        return (image_tensor, torch.tensor([0, 1, 2, 3]))

    def __len__(self):
        return self.data_len

#dataset = Data('images/')
#print(len(dataset))
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
#print(len(dataloader))
#for data in dataloader:
#    inputs, labels = data
#    print("Input batch shape:", inputs.shape)
#    print("Label batch shape:", labels.shape)
#    break