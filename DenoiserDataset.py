import torch.utils.data as data
import torchvision
from torchvision import transforms
import os
import random

import numpy as np
from os import listdir
from os.path import join
import cv2
from PIL import  Image

from DeepImageDenoiser import DIMENSION, IMAGE_SIZE

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_image(filepath, channels):
    if channels == 3:
        image = Image.open(filepath).convert('RGB')
    else :
        image = Image.open(filepath).convert('L')
    #image = image.resize((300 , 300), Image.BICUBIC)
    return image

def denoise(image):
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    img = np.array(image)
    out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    result = Image.fromarray(out)

    if DIMENSION == 1:
        result = result.convert('L')

    return result


class DenoiserDataset(data.Dataset):
    def __init__(self, image_dir, augmentation: bool = False):
        super(DenoiserDataset, self).__init__()
        self.deprocess = False
        self.augmentation = None

        if augmentation:
            self.augmentation = transforms.Compose([
                                  transforms.RandomCrop(IMAGE_SIZE),
                                  transforms.RandomHorizontalFlip(),
            ])

        self.tensoration =  transforms.Compose([
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.images = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.distorter = transforms.Compose([RandomNoise()])

    def __getitem__(self, index):
        target = load_image(self.images[index], DIMENSION)

        if  self.augmentation is not None:
            target = self.augmentation(target)

        distorted = self.distorter (target)
        input, target = self.tensoration(distorted), self.tensoration(target)
        if self.deprocess:
            denoised = denoise(distorted)
            return input, target, self.tensoration(denoised)
        else:
            return input, target

    def __len__(self):
        return len(self.images)


class RandomNoise(object):
    def __init__(self):
        self.sigma = 15
        self.random_state= np.random.RandomState(42)

    def __call__(self, img):
        transforms = []
        noise = np.random.choice([1, 2 ])
        noise = 2
        if noise == 1:
            transforms.append(Lambda(lambda img: self.camera_noise(img, self.sigma)))
        else:
            transforms.append(Lambda(lambda img: self.gaussian_noise(img, self.sigma)))

        transform = torchvision.transforms.Compose(transforms)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ', sigma={0}'.format(self.sigma)
        return format_string

    def camera_noise(self, input, sigma):
        if sigma > 0:
            img = np.array(input)
            img = img.astype(dtype=np.float32)
            photons = self.random_state.poisson(img, size=img.shape)
            electrons = 0.69 * photons
            additive_noise = self.random_state.normal(scale=sigma, size=electrons.shape)
            additive_noise = cv2.GaussianBlur(additive_noise,(0,0), 0.3)
            noisy_img =  electrons + additive_noise
            noisy_img  = (noisy_img * 1.33).astype(np.int)
            noisy_img += 6
            noisy_img = np.clip(noisy_img, 0, 255)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            return Image.fromarray(noisy_img)
        else:
            return input

    def gaussian_noise(self, input, sigma):
        if sigma > 0:
            input = np.array(input)
            img = input.astype(dtype=np.float32)
            noisy_img = img + np.random.normal(0.0, sigma, img.shape)
            noisy_img = np.clip(noisy_img, 0.0, 255.0)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            return Image.fromarray(noisy_img)
        else:
            return input


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
