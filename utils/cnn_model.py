import torch.nn as nn
from torchvision import models


class CNNNet(nn.Module):
    def __init__(self, model_name, code_length, pretrained=True):
        super(CNNNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg11'
        if model_name == 'resnet50':
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet50'

        if model_name == 'multihead':
            code_len, code_len1, code_len2 = code_length
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_len),
                nn.Tanh()
            )
            self.classifier1 = nn.Sequential(
                nn.Linear(2048, code_len1),
                nn.Tanh()
            )
            self.classifier2 = nn.Sequential(
                nn.Linear(2048, code_len2),
                nn.Tanh()
            )
            self.model_name = 'multihead'
        if model_name == 'cascade':
            code_len, code_len1, code_len2 = code_length
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.linear = nn.Linear(2048, code_len2)
            self.linear1 = nn.Linear(code_len2, code_len1)
            self.linear2 = nn.Linear(code_len1, code_len)
            self.tanh = nn.Tanh()
            self.model_name = 'cascade'

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
            y = self.classifier(f)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
            y = self.classifier(f)
        if self.model_name == 'resnet50':
            f = f.view(f.size(0), -1)
            y = self.classifier(f)
        if self.model_name == 'multihead':
            f = f.view(f.size(0), -1)
            y = self.classifier(f)
            y1 = self.classifier1(f)
            y2 = self.classifier2(f)
            return y, y1, y2
        if self.model_name == 'cascade':
            f = f.view(f.size(0), -1)
            y2 = self.linear(f)
            y1 = self.linear1(y2)
            y = self.linear2(y1)
            return self.tanh(y), self.tanh(y1), self.tanh(y2)
        return y

class CrossNet(nn.Module):
    def __init__(self, model_name, code_length):
        super(CrossNet, self).__init__()

        if model_name == 'cross_net':
            self.img_classifier = nn.Sequential(
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.txt_classifier = nn.Sequential(
                nn.Linear(5000, code_length),
                nn.Tanh()
            )
            self.model_name = 'cross_net'

    def forward(self, x, y):
        x_code = self.img_classifier(x)
        y_code = self.txt_classifier(y)
        return x_code, y_code

class CNNExtractNet(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CNNExtractNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
            )
            self.model_name = 'vgg11'
        if model_name == "resnet50":
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'resnet50'


    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)

        if self.model_name == "resnet50":
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
