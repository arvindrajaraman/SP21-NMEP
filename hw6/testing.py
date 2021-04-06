from resnet import ResNet
import torch
import matplotlib
import numpy as np
from PIL import Image
import PIL

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dict = unpickle('cifar-10-batches-py/test_batch')
images = data_dict[b'data'].reshape(10000, 3, 32, 32).astype("uint8")
accuracy = 0.0

model = ResNet(None, None)
model.load_state_dict(torch.load('rotationnetmodelbest.pth.tar'))
model.eval()
num_samples = 32
for i in range(num_samples):
    image_PIL = PIL.Image.fromarray(np.uint8(images[i].transpose(1, 2, 0)))
    test_batch = torch.tensor(np.stack([np.asarray(image_PIL.rotate(0)).transpose(2, 0, 1) / 255,
                                        np.asarray(image_PIL.rotate(90)).transpose(2, 0, 1) / 255,
                                        np.asarray(image_PIL.rotate(180)).transpose(2, 0, 1) / 255,
                                        np.asarray(image_PIL.rotate(270)).transpose(2, 0, 1) / 255])).float()
    # print(test_batch.shape)
    accuracy += (torch.argmax(model(test_batch), dim=1) == torch.tensor([0.0, 1.0, 2.0, 3.0])).float().sum()
print("Accuracy: ", accuracy / (num_samples * 4))
print("done")
