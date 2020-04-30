from torch import nn
from torchvision.models.densenet import _DenseLayer, _DenseBlock


class DenseLayer1D(_DenseLayer):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4,
                 drop_rate=0, kernel_size=3,
                 memory_efficient=False):
        """
        Current __init__ is copy, paste, edit from
        https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

        Parameters
        ----------
        num_input_features
        growth_rate
        bn_size
        drop_rate
        memory_efficient
        """

        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1,
                                           stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=(1, kernel_size),
                                           stride=1,
                                           padding=(0, 1),
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient


class DenseBlock1D(_DenseBlock):

    def __init__(self, num_layers, num_input_features, bn_size=4,
                 growth_rate=32, kernel_size=3,
                 drop_rate=0, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer1D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                kernel_size=kernel_size,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

