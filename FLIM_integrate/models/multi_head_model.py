import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, cnn_list, num_class):
        super().__init__()
        self.features_list = nn.ModuleList(cnn_list)

        self.classifier = nn.Sequential(
            nn.Linear(2048*len(cnn_list), 4096),
            # nn.Linear(512*len(cnn_list), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        features_output_list = []
        # Feature map average
        # for features in self.features_list:
        #     features_output = features(x)
        #     features_output_list.append(features_output)
        # output = torch.mean(torch.stack(features_output_list, dim=0), dim=0)
        
        # output = output.view(output.size()[0], -1)
        
        # Feature map concat
        for features in self.features_list:
            features_output = features(x)
            features_output_list.append(features_output.view(features_output.size(0), -1))
        output = torch.cat(features_output_list, dim=1)        

        output = self.classifier(output)

        return output
    
    def forward_softmax(self, x):
        output = self.forward(x)
        return F.softmax(output, dim=1)

    

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def multi_head_vgg11_bn(n_header):
    cnn_header_list = []
    for _ in range(n_header):
        header = make_layers(cfg['A'], batch_norm=True)
        cnn_header_list.append(header)
    
    return VGG(cnn_header_list, n_header)




