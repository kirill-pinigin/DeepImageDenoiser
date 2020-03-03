import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from DeepImageDenoiser import  LR_THRESHOLD, DIMENSION, LEARNING_RATE
from NeuralModels import  SpectralNorm

ITERATION_LIMIT = int(1e6)
SQUEEZENET_CONFIG = {'dnn' : models.squeezenet1_1(pretrained=True).features, 'features' :  [2, 5, 8,  13]}
VGG_16_CONFIG = {'dnn' : models.vgg16(pretrained=True).features, 'features' :  [4, 9, 16,  23]}
VGG_16_BN_CONFIG = {'dnn' : models.vgg16_bn(pretrained=True).features, 'features' :  [6, 13, 23, 33] }
VGG_19_CONFIG = {'dnn' : models.vgg19(pretrained=True).features, 'features' : [ 4,  9, 18, 36] }
VGG_19_BN_CONFIG = {'dnn': models.vgg19_bn(pretrained=True).features, 'features' : [6, 13, 23, 52]}


class BasicFeatureExtractor(nn.Module):
    def __init__(self,  vgg_config , feature_limit = 9):
        super(BasicFeatureExtractor, self).__init__()
        if DIMENSION == 3:
            self.mean = Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
            self.std = Parameter(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

        elif DIMENSION == 1:
            self.mean = Parameter(torch.tensor([0.449]).view(-1, 1, 1))
            self.std = Parameter(torch.tensor([0.226]).view(-1, 1, 1))

        else:
            self.mean = Parameter(torch.zeros(DIMENSION).view(-1, 1, 1))
            self.std = Parameter(torch.ones(DIMENSION).view(-1, 1, 1))

        vgg_pretrained = vgg_config['dnn']
        conv = BasicFeatureExtractor.configure_input(DIMENSION, vgg_pretrained)

        self.slice1 = nn.Sequential(conv)
        for x in range(1, feature_limit):
            self.slice1.add_module(str(x), vgg_pretrained[x])

    @staticmethod
    def configure_input(dimension, vgg):
        conv = nn.Conv2d(dimension, 64, kernel_size=3, padding=1)

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, DIMENSION, 3, 3)
            parameters = list(vgg.parameters())
            for i in range(64):
                if DIMENSION == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        return conv

    def forward(self, x):
        if DIMENSION == 1 or DIMENSION == 3:
            if self.mean.device != x.device:
                self.mean.to(x.device)

            if self.std.device != x.device:
                self.std.to(x.device)

            x = (x - self.mean) / self.std

        return self.slice1(x)


class BasicMultiFeatureExtractor(BasicFeatureExtractor):
    def __init__(self,  vgg_config , requires_grad):
        super(BasicMultiFeatureExtractor, self).__init__(vgg_config, vgg_config['features'][0])
        vgg_pretrained = vgg_config['dnn']

        self.slice2 = torch.nn.Sequential()
        for x in range(vgg_config['features'][0], vgg_config['features'][1]):
            self.slice2.add_module(str(x), vgg_pretrained[x])

        self.slice3 = torch.nn.Sequential()
        for x in range(vgg_config['features'][1], vgg_config['features'][2]):
            self.slice3.add_module(str(x), vgg_pretrained[x])

        self.slice4 = torch.nn.Sequential()
        for x in range(vgg_config['features'][2], vgg_config['features'][3]):
            self.slice4.add_module(str(x), vgg_pretrained[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = super(BasicMultiFeatureExtractor, self).forward(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        return h_relu1, h_relu2, h_relu3, h_relu4


class FastNeuralStyleExtractor(BasicMultiFeatureExtractor):
    def __init__(self, requires_grad=False , bn = True):
        features = VGG_16_BN_CONFIG if bn else VGG_16_CONFIG
        super(FastNeuralStyleExtractor, self).__init__(features, requires_grad)


class FastNeuralStylePerceptualLoss(nn.Module):
    def __init__(self, weight:float = 1e-3):
        super(FastNeuralStylePerceptualLoss, self).__init__()
        self.factors = [1e0 , 1e-1, 1e-2 , 1e-3]
        self.weight = weight
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.features = FastNeuralStyleExtractor()
        self.features.eval()
        self.features.to(self.device)
        self.criterion = nn.MSELoss()

    def compute_gram_matrix(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, actual, desire):
        actuals = torch.nn.parallel.data_parallel(module=self.features, inputs=actual, device_ids=self.cudas)
        desires = torch.nn.parallel.data_parallel(module=self.features, inputs=desire, device_ids=self.cudas)

        closs = 0.0
        for i in range(len(actuals)):
            closs +=  self.factors[i] * self.criterion(actuals[i], desires[i])

        sloss = 0.0
        if self.weight != 0:
            self.weight * self.criterion(self.compute_gram_matrix(actuals[i]),
                                         self.compute_gram_matrix(desires[i]))

        self.loss = closs + sloss
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)


class FluentExtractor(BasicMultiFeatureExtractor):
    def __init__(self):
        super(BasicFeatureExtractor, self).__init__()
        self.mean = Parameter(torch.zeros(DIMENSION).view(-1, 1, 1))
        self.std = Parameter(torch.ones(DIMENSION).view(-1, 1, 1))

        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=DIMENSION, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.slice2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.slice3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.slice4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )


class AdaptivePerceptualLoss(nn.Module):
    def __init__(self):
        super(AdaptivePerceptualLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.features = FluentExtractor()
        self.factors = [1e0, 1e-1, 1e-2, 1e-3]
        self.predictor = nn.Sequential()
        self.predictor.add_module('conv_9', nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False))
        self.predictor.add_module('lrelu_9', nn.LeakyReLU(0.2))
        self.predictor.add_module('fc', nn.Conv2d(8, 1, 1, 1, 0, bias=False))
        self.predictor.add_module('sigmoid', nn.Sigmoid())
        self.features.to(self.device)
        self.predictor.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.ContentCriterion = nn.L1Loss()
        self.AdversarialCriterion = nn.BCELoss()
        self.loss = None
        self.counter = int(0)
        self.best_loss = float(100500)
        self.current_loss = float(0)
        self.relu = nn.ReLU()
        self.margin = 1.0

    def evaluate(self, actual, desire):
        actual_features = torch.nn.parallel.data_parallel(module=self.features, inputs=actual, device_ids=self.cudas)
        desire_features = torch.nn.parallel.data_parallel(module=self.features, inputs=desire, device_ids=self.cudas)
        ploss = 0.0

        for i in range(len(desire_features)):
            ploss += self.factors[i]*self.ContentCriterion(actual_features[i], desire_features[i])

        return actual_features, desire_features, ploss

    def meta_optimize(self, lossD, length):
        self.current_loss += float(lossD.item()) / length

        if self.counter > ITERATION_LIMIT:
            self.current_loss = self.current_loss / float(ITERATION_LIMIT)
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                print('! best_loss !', self.best_loss)
            else:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.2
                        print('! Decrease LearningRate in Perceptual !', lr)
            self.counter = int(0)
            self.current_loss = float(0)

        self.counter += int(1)

    def pretrain(self, dataloaders, num_epochs=20):
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.features.train(True)
                    self.predictor.train(True)
                else:
                    self.features.train(False)
                    self.predictor.train(False)

                running_loss = 0.0
                running_corrects = 0
                for data in dataloaders[phase]:
                    inputs, targets = data
                    targets = targets.float()
                    inputs = Variable(inputs.to(self.device))
                    targets = Variable(targets.to(self.device))
                    self.optimizer.zero_grad()
                    features = torch.nn.parallel.data_parallel(module=self.features, inputs=inputs, device_ids=self.cudas)
                    outputs = torch.nn.parallel.data_parallel(module=self.predictor, inputs=features[-1].detach(), device_ids=self.cudas).view(-1)
                    loss = self.AdversarialCriterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                    running_corrects += torch.sum(torch.round(outputs.data) == targets.data)
                    self.meta_optimize(loss, float(targets.size(0)))

                epoch_loss = float(running_loss) / float(len(dataloaders[phase].dataset))
                epoch_acc = float(running_corrects) / float(len(dataloaders[phase].dataset))
                print(' epoch_acc ', epoch_acc, ' epoch_loss ', epoch_loss)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print('curent best_acc ', best_acc)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def fit(self, actual, desire):
        self.features.train()
        self.predictor.train()
        self.optimizer.zero_grad()
        actual_features, desire_features, ploss = self.evaluate(actual, desire)
        fake = torch.nn.parallel.data_parallel(module=self.predictor, inputs=actual_features[-1].detach(),device_ids=self.cudas).view(-1)
        zeros = Variable(torch.zeros(fake.shape).to(self.device))
        real = torch.nn.parallel.data_parallel(module=self.predictor, inputs=desire_features[-1].detach(), device_ids=self.cudas).view(-1)
        ones = Variable(torch.ones(real.shape).to(self.device))
        lossDreal = self.AdversarialCriterion(real, ones)
        lossDfake = self.AdversarialCriterion(fake, zeros)
        lossD = lossDreal + lossDfake + self.relu(self.margin - ploss).mean()
        lossD.backward(retain_graph=True)
        self.optimizer.step()
        self.meta_optimize(lossD, float(actual.size(0)))

    def forward(self, actual, desire):
        self.predictor.eval()
        self.features.eval()
        actual_features, _, ploss = self.evaluate(actual, desire)
        rest = self.predictor(actual_features[-1]).view(-1)
        ones = Variable(torch.ones(rest.shape).to(self.device))
        aloss = self.AdversarialCriterion(rest, ones)
        self.loss = ploss + aloss + self.ContentCriterion(actual, desire)
        self.fit(actual, desire)
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)


class MobileExtractor(BasicMultiFeatureExtractor):
    def __init__(self, requires_grad=False, bn = True):
        features = VGG_19_BN_CONFIG if bn else VGG_19_CONFIG
        super(MobileExtractor, self).__init__(features, requires_grad)


class MobilePerceptualLoss(nn.Module):
    def __init__(self):
        super(MobilePerceptualLoss, self).__init__()
        self.factors = [1e0, 1e-1, 1e-2, 1e-3]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.features = MobileExtractor()
        self.features.eval()
        self.features.to(self.device)
        self.criterion = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = torch.nn.parallel.data_parallel(module=self.features, inputs=actual, device_ids=self.cudas)
        desires = torch.nn.parallel.data_parallel(module=self.features, inputs=desire, device_ids=self.cudas)

        loss = 0.0
        for i in range(len(actuals)):
            loss +=  self.factors[i]*self.criterion(actuals[i], desires[i])

        self.loss = loss
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)


class SimpleExtractor(BasicFeatureExtractor):
    def __init__(self, feat=1, bn = True):
        features_list = VGG_19_BN_CONFIG['features'] if bn else VGG_19_CONFIG['features']
        features_limit = features_list[1]
        super(SimpleExtractor, self).__init__(VGG_19_CONFIG, features_limit)


class SimplePerceptualLoss(nn.Module):
    def __init__(self, feat  : int = 2):
        super(SimplePerceptualLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.features = SimpleExtractor(feat)
        self.features.eval()
        self.features.to(self.device)
        self.criterion = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = torch.nn.parallel.data_parallel(module=self.features, inputs=actual, device_ids=self.cudas)
        desires = torch.nn.parallel.data_parallel(module=self.features, inputs=desire, device_ids=self.cudas)
        loss = self.criterion(actuals, desires)
        self.loss = loss
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)


class SqueezeExtractor(BasicMultiFeatureExtractor):
    def __init__(self, requires_grad=False):
        super(SqueezeExtractor, self).__init__(SQUEEZENET_CONFIG, requires_grad)


class SqueezeAdaptivePerceptualLoss(AdaptivePerceptualLoss):
    def __init__(self):
        super(SqueezeAdaptivePerceptualLoss, self).__init__()
        self.features = SqueezeExtractor(requires_grad=True)
        self.features.to(self.device)
        self.predictor.to(self.device)


class SpectralFluentExtractor(BasicMultiFeatureExtractor):
    def __init__(self):
        super(BasicFeatureExtractor, self).__init__()
        self.mean = Parameter(torch.zeros(DIMENSION).view(-1, 1, 1))
        self.std = Parameter(torch.ones(DIMENSION).view(-1, 1, 1))

        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=DIMENSION, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.slice2 = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.slice3 = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.slice4 = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )


class SpectralAdaptivePerceptualLoss(AdaptivePerceptualLoss):
    def __init__(self):
        super(SpectralAdaptivePerceptualLoss, self).__init__()
        self.features = SpectralFluentExtractor()
        self.predictor = nn.Sequential()
        self.predictor.add_module('fc', SpectralNorm(nn.Conv2d(8, 1, 1, 1, 0, bias=False)))
        self.features.to(self.device)
        self.predictor.to(self.device)

    def fit(self, actual, desire):
        self.features.train()
        self.predictor.train()
        self.optimizer.zero_grad()
        actual_features, desire_features, ploss = self.evaluate(actual, desire)
        fake = torch.nn.parallel.data_parallel(module=self.predictor, inputs=actual_features[-1].detach(),
                                               device_ids=self.cudas).view(-1)
        real = torch.nn.parallel.data_parallel(module=self.predictor, inputs=desire_features[-1].detach(),
                                               device_ids=self.cudas).view(-1)
        lossDreal = self.relu(1.0 - real).mean()
        lossDfake = self.relu(1.0 + fake).mean()

        lossD = lossDreal + lossDfake + self.relu(self.margin - ploss).mean()
        lossD.backward(retain_graph=True)
        self.optimizer.step()
        self.meta_optimize(lossD, float(actual.size(0)))

    def forward(self, actual, desire):
        self.predictor.eval()
        self.features.eval()
        actual_features, _, ploss = self.evaluate(actual, desire)
        self.loss = ploss - self.predictor(actual_features[-1]).view(-1).mean() + self.ContentCriterion(actual, desire)
        self.fit(actual, desire)
        return self.loss


class WassersteinAdaptivePerceptualLoss(SpectralAdaptivePerceptualLoss):
    def __init__(self):
        super(WassersteinAdaptivePerceptualLoss, self).__init__()
        self.predictor.add_module('sigmoid', nn.Sigmoid())
        self.predictor.to(self.device)

    def forward(self, actual, desire):
        self.predictor.eval()
        self.features.eval()
        actual_features, _, ploss = self.evaluate(actual, desire)
        result = self.predictor(actual_features[-1]).view(-1)
        self.loss = ploss - result.view(-1).mean() + torch.nn.functional.binary_cross_entropy(result, torch.ones_like(result))
        self.fit(actual, desire)
        return self.loss

    def fit(self, actual, desire):
        self.features.train()
        self.predictor.train()
        self.optimizer.zero_grad()
        actual_features, desire_features, ploss = self.evaluate(actual, desire)
        fake = torch.nn.parallel.data_parallel(module=self.predictor, inputs=actual_features[-1].detach(),
                                               device_ids=self.cudas).view(-1)
        real = torch.nn.parallel.data_parallel(module=self.predictor, inputs=desire_features[-1].detach(),
                                               device_ids=self.cudas).view(-1)
        real_loss = torch.nn.functional.binary_cross_entropy(real, Variable(torch.ones_like(real)).to(self.device))
        fake_loss = torch.nn.functional.binary_cross_entropy(fake, Variable(torch.zeros_like(fake)).to(self.device))
        wgan_loss = fake.mean() - real.mean()
        interpolates  = 0.5 * desire + (1 - 0.5) * actual
        interpolates = Variable(interpolates.clone(), requires_grad=True).to(self.device)
        interpolatesl_features = torch.nn.parallel.data_parallel(module=self.features, inputs=interpolates, device_ids=self.cudas)
        interpolates_discriminator_out  = torch.nn.parallel.data_parallel(module=self.predictor, inputs=interpolatesl_features[-1], device_ids=self.cudas).view(-1)

        buffer = Variable(torch.ones_like(interpolates_discriminator_out), requires_grad=True).to(self.device)
        gradients = torch.autograd.grad(outputs=interpolates_discriminator_out, inputs=interpolates,
                                  grad_outputs=buffer,
                                  retain_graph=True,
                                  create_graph=True)[0]

        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        lossD =  (real_loss + fake_loss) / 2.0 + wgan_loss + 1e-2*gradient_penalty
        lossD.backward()
        self.optimizer.step()
        self.current_loss += float(lossD.item()) / float(actual.size(0))

        if self.counter > ITERATION_LIMIT:
            self.current_loss = self.current_loss / float(ITERATION_LIMIT)
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                print('! best_loss !', self.best_loss)
            else:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.2
                        print('! Decrease LearningRate in Perceptual !', lr)
            self.counter = int(0)
            self.current_loss = float(0)

        self.counter += int(1)
