import numpy as np
from PIL import Image
import PIL
import glob
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

image_files = glob.glob('hw6/cifar-10-batches-py/data_batch_?')
# image_list = np.array([])
for file in image_files:
    data_dict = unpickle(file)
    images = data_dict[b'data'].reshape(10000, 3, 32, 32).astype("uint8")
    filenames = data_dict[b'filenames']
    for i in range(len(images)):
        image_PIL = PIL.Image.fromarray(np.uint8(images[i].transpose(1, 2, 0)))
        image_PIL.save('hw6/images/' + filenames[i].decode("utf-8"))
print("done")

print(len(os.listdir('hw6/images/')))