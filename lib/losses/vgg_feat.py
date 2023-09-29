import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # shift to [0 1] before normalization
    batch = batch/2. + 0.5
    return (batch - mean) / std

# VGG architecter, used for the perceptual-like loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        try:
            # if with internet
            vgg19_net = torchvision.models.vgg19(pretrained=True)
        except:
            # without internet, load from cache
            vgg19_net = torchvision.models.vgg19(pretrained=False)
            vgg19_net.load_state_dict(
                torch.load("./cache/pretrained_models/vgg19-dcbb9e9d.pth")
            )
        vgg_pretrained_features = vgg19_net.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, mask=None):
        B, C, H, W = X.shape
        X = normalize_batch(X[:, :3])
        if mask is not None:
            X = X * mask # apply mask after normalization
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [F.interpolate(h_relu3, [H//2, W//2], mode='bilinear'), 
        #        F.interpolate(h_relu4, [H//2, W//2], mode='bilinear')]
        # out = torch.cat(out, dim=1)
        return h_relu3