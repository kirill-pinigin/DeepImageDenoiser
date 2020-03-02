import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn

from AdversarialLoss import MobileImprovingAdversarialLoss,MultiSigmaAdversarialLoss, PhotoRealisticAdversarialLoss,  WassersteinAdversarialLoss
from DeepImageDenoiser import DeepImageDenoiser , LEARNING_RATE
from DenoiserDataset import  DenoiserDataset
from RSGUNetGenerator import RSGUNetGenerator
from NeuralModels import SILU, UpsampleDeConv, TransposedDeConv, PixelDeConv
from PerceptualLoss import AdaptivePerceptualLoss, FastNeuralStylePerceptualLoss, SpectralAdaptivePerceptualLoss, WassersteinAdaptivePerceptualLoss  , SimplePerceptualLoss, SqueezeAdaptivePerceptualLoss
from ResidualGenerator import ResidualGenerator
from UNetGenerator import UNetGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir',         type = str,   default='./BSDS500/', help='path to dataset')
parser.add_argument('--generator',         type = str,   default='RSGUNet', help='type of image generator')
parser.add_argument('--criterion',         type = str,   default='MobileImproving', help='type of criterion')
parser.add_argument('--deconv',            type = str,   default='Upsample', help='type of deconv')
parser.add_argument('--activation',        type = str,   default='Leaky', help='type of activation')
parser.add_argument('--optimizer',         type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--batch_size',        type = int,   default=32)
parser.add_argument('--epochs',            type = int,   default=128)
parser.add_argument('--resume_train',      type = bool,  default=True)

args = parser.parse_args()

print(torch.__version__)

criterion_types =   {
                        'MobileImproving'       : MobileImprovingAdversarialLoss(),
                        'MultiSigma'            : MultiSigmaAdversarialLoss(),
                        'PhotoRealistic'        : PhotoRealisticAdversarialLoss(),
                        'Wasserstein'           : WassersteinAdversarialLoss(),
                        'FastNeuralStyle'       : FastNeuralStylePerceptualLoss(),
                        'PAN'                   : AdaptivePerceptualLoss(),
                        'SimplePerceptual'      : SimplePerceptualLoss(),
                        'SpectralPAN'           : SpectralAdaptivePerceptualLoss(),
                        'SqueezeAdaptive'       : SqueezeAdaptivePerceptualLoss(),
                        'WassersteinAdaptive'   : WassersteinAdaptivePerceptualLoss(),
                        'MSE'                   : nn.MSELoss(),
                    }

generator_types = {
                        'UNet'     : UNetGenerator,
                        'RSGUNet'  : RSGUNetGenerator,
                        'Residual' : ResidualGenerator,
                    }

deconv_types =      {
                        'Transposed'  : TransposedDeConv,
                        'Upsample'    : UpsampleDeConv,
                        'Pixel'       : PixelDeConv
                    }

activation_types =  {
                        'ReLU' : nn.ReLU(),
                        'Leaky': nn.LeakyReLU(),
                        'PReLU': nn.PReLU(),
                        'ELU'  : nn.ELU(),
                        'SELU' : nn.SELU(),
                        'SILU' : SILU()
                    }

optimizer_types =   {
                        'Adam'   : optim.Adam,
                        'RMSprop': optim.RMSprop,
                        'SGD'    : optim.SGD
                    }

model = generator_types[args.generator]
deconvLayer = (deconv_types[args.deconv] if args.deconv in deconv_types else deconv_types['upsample'])
function = (activation_types[args.activation] if args.activation in activation_types else activation_types['Leaky'])
generator = model(deconv=deconvLayer, activation=function)
optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(generator.parameters(), lr = LEARNING_RATE)
criterion = criterion_types[args.criterion]

augmentations = {'train' : True, 'val' : False}
shufles = {'train' : True, 'val' : False}
batch_sizes = {'train' : args.batch_size, 'val' : args.batch_size if args.batch_size < 8 else 8}

image_datasets = {x: DenoiserDataset(os.path.join(args.image_dir, x),  augmentation = augmentations[x])
                    for x in ['train', 'val']}

imageloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                             shuffle=shufles[x], num_workers=4)
                for x in ['train', 'val']}

test_dataset = DenoiserDataset(args.image_dir+'/test/',  augmentation = False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

framework = DeepImageDenoiser(generator = generator, criterion = criterion, optimizer = optimizer)
framework.approximate(dataloaders = imageloaders, num_epochs=args.epochs, resume_train=args.resume_train)
framework.estimate(testloader)
