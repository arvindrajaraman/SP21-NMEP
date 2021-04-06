import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data.sampler import SubsetRandomSampler
from resnet import ResNet
from data import Data
import time
import shutil
import yaml
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (input, target) in enumerate(train_loader, start=1):
        rotations = torch.split(input, split_size_or_sections=3, dim=1)
        minibatch = torch.cat([rotations[0], rotations[1], rotations[2], rotations[3]])
        target = target.view(-1)
        targets = torch.cat([target[::4], target[1::4], target[2::4], target[3::4]])

        outputs = model(minibatch)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print("Epoch: {0:03d} | Training Loss: {1:.5f}".format(epoch, running_loss / i))


def validate(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0
    avg_acc = 0.0
    for i, (input, target) in enumerate(val_loader, start=1):
    	# TODO: implement the validation. Remember this is validation and not training
    	# so some things will be different.
        rotations = torch.split(input, split_size_or_sections=3, dim=1)
        minibatch = torch.cat([rotations[0], rotations[1], rotations[2], rotations[3]])
        target = target.view(-1)
        targets = torch.cat([target[::4], target[1::4], target[2::4], target[3::4]])

        outputs = model(minibatch)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        avg_acc += (torch.argmax(outputs, dim=1) == targets).float().sum() / len(targets)
    print("Epoch: {0:03d} | Validation Loss: {1:.5f} | Validation Accuracy: {2:.5f}".format(epoch, running_loss / i, avg_acc / i))


def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    # best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
    n_epochs = config["num_epochs"]
    model = ResNet(None, None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    
    dataset = Data('images/')
    indices = list(range(50000))
    np.random.shuffle(indices)
    train_indices, vali_indices = train_indices, val_indices = indices[:10240], indices[10240:12288]
    train_dataset = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_dataset)
    val_dataset = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=val_dataset)

    for epoch in range(n_epochs):
        # TODO: make your loop which trains and validates. Use the train() func
        train(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion, epoch)
        # TODO: Save your checkpoint
        save_checkpoint(model.state_dict(), True)

if __name__ == "__main__":
    main()
