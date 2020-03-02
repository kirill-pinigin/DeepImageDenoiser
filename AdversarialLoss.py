import torch
import torch.nn as nn

from torch.autograd import Variable

from DeepImageDenoiser import  LR_THRESHOLD, DIMENSION, LEARNING_RATE
from NeuralModels import  SpectralNorm, TotalVariation
from PerceptualLoss import  FastNeuralStylePerceptualLoss , SimplePerceptualLoss, MobilePerceptualLoss
from SSIM import SSIMLoss
ITERATION_LIMIT = int(1e6)


class AdversarialLoss(nn.Module):
    def __init__(self, weight : float = 1e-3):
        super(AdversarialLoss, self).__init__()
        self.weight = weight
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=DIMENSION, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.discriminator.to(self.device)

        self.ContentCriterion = nn.MSELoss()
        self.AdversarialCriterion = nn.BCELoss()
        self.loss = None
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)
        self.counter = int(0)
        self.best_loss = float(100500)
        self.current_loss = float(0)


    def forward(self, actual, desire):
        self.discriminator.eval()
        rest = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=actual, device_ids=self.cudas).view(-1)
        ones = Variable(torch.ones(rest.shape).to(self.device))
        self.loss = self.ContentCriterion(actual, desire) +  self.weight * self.AdversarialCriterion(rest, ones)
        self.fit(actual, desire)
        return self.loss

    def fit(self, actual, desire):
        self.discriminator.train()
        self.optimizer.zero_grad()
        real = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=desire.detach(), device_ids=self.cudas).view(-1)
        ones = Variable(torch.ones(real.shape).to(self.device))
        fake = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=actual.detach(), device_ids=self.cudas).view(-1)
        zeros = Variable(torch.zeros(fake.shape).to(self.device))
        lossDreal = self.AdversarialCriterion(real, ones)
        lossDfake = self.AdversarialCriterion(fake, zeros)
        lossD = lossDreal + lossDfake
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

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)


class MobileImprovingAdversarialLoss(AdversarialLoss):
    def __init__(self, weight : float = 1e-3):
        super(MobileImprovingAdversarialLoss, self).__init__(weight)
        self.ContentCriterion = MobilePerceptualLoss()
        self.ssim = SSIMLoss()
        self.tv = TotalVariation()

    def forward(self, actual, desire):
        self.loss = super(MobileImprovingAdversarialLoss, self).forward(actual, desire) + self.ssim(actual, desire)+ self.tv(actual) + nn.functional.l1_loss(actual, desire)
        return self.loss


class MultiSigmaAdversarialLoss(AdversarialLoss):
    def __init__(self, weight : float = 1e-3):
        super(MultiSigmaAdversarialLoss, self).__init__(weight)
        self.ContentCriterion = FastNeuralStylePerceptualLoss(weight)

    def forward(self, actual, desire):
        self.loss = super(MultiSigmaAdversarialLoss, self).forward(actual, desire) + nn.functional.mse_loss(actual, desire)
        return self.loss


class PhotoRealisticAdversarialLoss(AdversarialLoss):
    def __init__(self, weight : float = 1e-3):
        super(PhotoRealisticAdversarialLoss, self).__init__(weight)
        self.ContentCriterion = SimplePerceptualLoss('feat2_2')
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=DIMENSION, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.discriminator.to(self.device)


class WassersteinAdversarialLoss(AdversarialLoss):
    def __init__(self):
        super(WassersteinAdversarialLoss, self).__init__()
        self.ContentCriterion = SimplePerceptualLoss('feat3_3')

    def forward(self, actual, desire):
        self.discriminator.eval()
        result = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=actual, device_ids=self.cudas)
        self.loss = 100*self.ContentCriterion(actual, desire) - result.view(-1).mean() + torch.nn.functional.binary_cross_entropy(result, torch.ones_like(result))
        self.fit(actual, desire)
        return self.loss

    def fit(self, actual, desire):
        self.discriminator.train()
        self.optimizer.zero_grad()
        real = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=desire.detach(), device_ids=self.cudas).view(-1)
        fake = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=actual.detach(), device_ids=self.cudas).view(-1)
        real_loss = torch.nn.functional.binary_cross_entropy(real, Variable(torch.ones_like(real)).to(self.device))
        fake_loss = torch.nn.functional.binary_cross_entropy(fake, Variable(torch.zeros_like(fake)).to(self.device))
        wgan_loss = fake.mean() - real.mean()
        interpolates  = 0.5 * desire + (1 - 0.5) * actual
        interpolates = Variable(interpolates.clone(), requires_grad=True).to(self.device)
        interpolates_discriminator_out  = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=interpolates, device_ids=self.cudas).view(-1)

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


class SpectralAdversarialLoss(nn.Module):
    def __init__(self):
        super(SpectralAdversarialLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.ContentCriterion = SimplePerceptualLoss('feat3_3')
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=DIMENSION, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)),
        )

        self.relu = nn.ReLU()
        self.loss = None
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)
        self.counter = int(0)
        self.best_loss = float(100500)
        self.current_loss = float(0)
        self.discriminator.to(self.device)

    def forward(self, actual, desire):
        self.discriminator.eval()
        self.loss = 100*self.ContentCriterion(actual, desire) - torch.nn.parallel.data_parallel(module=self.discriminator, inputs=actual, device_ids=self.cudas).view(-1).mean()
        self.fit(actual, desire)
        return self.loss

    def fit(self, actual, desire):
        self.discriminator.train()
        self.optimizer.zero_grad()
        real = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=desire.detach(), device_ids=self.cudas).view(-1)
        fake = torch.nn.parallel.data_parallel(module=self.discriminator, inputs=actual.detach(), device_ids=self.cudas).view(-1)
        lossDreal = self.relu(1.0 - real).mean()
        lossDfake = self.relu(1.0 + fake).mean()
        lossD = lossDreal + lossDfake
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

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)
