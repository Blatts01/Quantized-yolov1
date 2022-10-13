from torch import nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat as SignedWeightQuant
from brevitas.quant import ShiftedUint8WeightPerTensorFloat as UnsignedWeightQuant
from brevitas.quant import ShiftedUint8ActPerTensorFloat as ActQuant
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant
from brevitas.export import PyXIRManager
from brevitas.quant import Int8Bias as BiasQuant

class ReducedRangeActQuant(ActQuant):
    bit_width = 7

class YOLOv1(nn.Module):
    """YOLOv1 model structure
    """

    def __init__(self, S, B, num_classes):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes

        # conv part
        self.conv_layers = nn.Sequential(
            # 448*448*3 -> 112*112*192
            nn.Conv2d(3, 192, 7, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 112*112*192 ->56*56*256
            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 56*56*256 -> 28*28*512
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 28*28*512 -> 14*14*1024
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 14*14*1024 -> 7*7*1024
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # 7*7*1024 -> 7*7*1024
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # full connection part
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.conv_layers(x)  # b*1024*7*7
        out = out.view(out.size()[0], -1)  # b*50176
        out = self.fc_layers(out)

        print(out.size())

        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)

        print(out.size())

        return out


class MixedFloatYolov1(nn.Module):
    def __init__(self, S, B, num_classes, weight_bits=8, activation_bits=8, in_bit_width=8):
        super(MixedFloatYolov1, self).__init__()

        self.S = S
        self.B = B
        self.num_classes = num_classes

        self.quant_inp = qnn.QuantIdentity(
            bit_width=in_bit_width, return_quant_tensor=True)

        # 448*448*3 -> 112*112*192
        self.conv1 = qnn.QuantConv2d( 
            3, 192, 7, stride=2, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        self.max_pool1 = nn.MaxPool2d(2, stride=2)

        # 112*112*192 ->56*56*256
        self.conv2 = qnn.QuantConv2d(
            192, 256, 1, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        self.max_pool2 = nn.MaxPool2d(2, stride=2)

        # 56*56*256 -> 28*28*512
        self.conv3 = qnn.QuantConv2d(
            256, 512, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        self.max_pool3 = nn.MaxPool2d(2, stride=2)

        # 28*28*512 -> 14*14*1024
        self.conv4 = qnn.QuantConv2d(
            512, 1024, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        self.max_pool4 = nn.MaxPool2d(2, stride=2)

        # 14*14*1024 -> 7*7*1024
        self.conv5 = qnn.QuantConv2d(
            1024, 1024, 3, stride=2, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu5 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)

        # 7*7*1024 -> 7*7*1024
        self.conv6 = qnn.QuantConv2d(
            1024, 1024, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu6 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=False)

        # full connection part
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            qnn.QuantReLU(bit_width=activation_bits, return_quant_tensor=False),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):

        out = self.quant_inp(x)

        out = self.relu1(self.conv1(out))
        out = self.max_pool1(out)

        out = self.relu2(self.conv2(out))
        out = self.max_pool2(out)

        out = self.relu3(self.conv3(out))
        out = self.max_pool3(out)

        out = self.relu4(self.conv4(out))
        out = self.max_pool4(out)

        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))

        out = out.view(out.size()[0], -1)  # b*50176
        out = self.fc_layers(out)

        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)

        return out




class QuantYOLOv1(nn.Module):
    # Quantized YOLOv1 model structure

    def __init__(self, S, B, num_classes, bias_quant=True, reduced_act_quant=False, weight_signed=False):
        super(QuantYOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes

        # conv part
        self.conv_layers = nn.Sequential(
            # 448*448*3 -> 112*112*192
            qnn.QuantConv2d(3, 192, 7, stride=2, padding=1, input_quant=act_quant, weight_quant=weight_quant,
            output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            #qnn.QuantReLU(return_quant_tensor=quant_relu),
            nn.MaxPool2d(2, stride=2),

            # 112*112*192 ->56*56*256
            qnn.QuantConv2d(192, 256, 3, padding=1,input_quant=act_quant, weight_quant=weight_quant,
            output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            #qnn.QuantReLU(return_quant_tensor=quant_relu, return_quant_tensor=quant_relu),
            nn.MaxPool2d(2, stride=2),

            # 56*56*256 -> 28*28*512
            qnn.QuantConv2d(256, 512, 3, padding=1,input_quant=act_quant, weight_quant=weight_quant,
            output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            #qnn.QuantReLU(return_quant_tensor=quant_relu),
            nn.MaxPool2d(2, stride=2),

            # 28*28*512 -> 14*14*1024
            qnn.QuantConv2d(512, 1024, 3, padding=1,input_quant=act_quant, weight_quant=weight_quant,
            output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            #qnn.QuantReLU(return_quant_tensor=quant_relu),
            nn.MaxPool2d(2, stride=2),

            # 14*14*1024 -> 7*7*1024
            qnn.QuantConv2d(1024, 1024, 3, stride=2, padding=1,input_quant=act_quant, weight_quant=weight_quant,
            output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            #qnn.QuantReLU(return_quant_tensor=quant_relu),

            # 7*7*1024 -> 7*7*1024
            qnn.QuantConv2d(1024, 1024, 3, padding=1,input_quant=act_quant, weight_quant=weight_quant,
            output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True),
            nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.1, inplace=True),
            qnn.QuantReLU(return_quant_tensor=True),
        )

        # full connection part
        self.fc_layers = nn.Sequential(
            qnn.QuantLinear(7 * 7 * 1024, 4096, bias=True, weight_quant=weight_quant,
            bias_quant=bias_quant, output_quant=act_quant),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.QuantReLU(return_quant_tensor=quant_relu),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.conv_layers(x)  # b*1024*7*7
        out = out.view(out.size()[0], -1)  # b*50176
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out


class MixedFloatYolov3(nn.Module):
    def __init__(self, S, B, num_classes, weight_bits=8, activation_bits=8, input_bits=8):
        super(MixedFloatYolov3, self).__init__()
        
        self.S = S
        self.B = B
        self.num_classes = num_classes

        self.quant_inp = qnn.QuantIdentity(
            bit_width=input_bits, return_quant_tensor=True)

        # QuantConv 416x416x3 416x416x8 3x3 1 QuantReLU
        self.conv1 = qnn.QuantConv2d(
            3, 8, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        
        # MaxPooling 416x416x8 208x208x8 2x2 2
        self.max_pool1 = nn.MaxPool2d(2,2)

        # QuantConv 208x208x8 208x208x8 3x3 1 QuantReLU
        self.conv2 = qnn.QuantConv2d(
            8, 8, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        
        # MaxPooling 208x208x8 104x104x8 2x2 2
        self.max_pool2 = nn.MaxPool2d(2,2)

        # QuantConv 104x104x8 104x104x16 3x3 1 QuantReLU
        self.conv3 = qnn.QuantConv2d(
            8, 16, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        
        # MaxPooling 104x104x16 52x52x16 2x2 2
        self.max_pool3 = nn.MaxPool2d(2,2)

        # QuantConv 52x52x16 52x52x32 3x3 1 QuantReLU
        self.conv4 = qnn.QuantConv2d(
            16, 32, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)

        # MaxPooling 52x52x32 26x26x32 2x2 2
        self.max_pool4 = nn.MaxPool2d(2,2)

        # QuantConv 26x26x32 26x26x56 3x3 1 QuantReLU
        self.conv5 = qnn.QuantConv2d(
            32, 56, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu5 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        
        # MaxPooling 26x26x56 13x13x56 2x2 2
        self.max_pool5 = nn.MaxPool2d(2,2)

        # QuantConv 13x13x56 13x13x104 3x3 1 QuantReLU
        self.conv6 = qnn.QuantConv2d(
            56, 104, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu6 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)
        
        # MaxPooling 13x13x104 13x13x104 2x2 2
        self.max_pool6 = nn.MaxPool2d(2,2)

        # QuantConv 13x13x104 13x13x208 3x3 1 QuantReLU
        self.conv7 = qnn.QuantConv2d(
            104, 208, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu7 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)

        # QuantConv 13x13x208 13x13x56 1x1 1 QuantReLU
        self.conv8 = qnn.QuantConv2d(
            208, 56, 1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu8 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)

        # QuantConv 13x13x56 13x13x104 3x3 1 QuantReLU
        self.conv9 = qnn.QuantConv2d(
            56, 104, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu9 = qnn.QuantReLU(
            bit_width=activation_bits, return_quant_tensor=True)

        # QuantConv 13x13x104 13x13x6 3x3 1
        self.conv10 = qnn.QuantConv2d(
            104, 12, 3, padding=1, weight_bit_width=weight_bits, bias_quant=BiasQuant, return_quant_tensor=True)

        #self.hard_tanh = qnn.QuantHardTanh(bit_width=activation_bits, min_val=-1, max_val=1, return_quant_tensor=True)
        self.hard_tanh = qnn.QuantHardTanh(bit_width=activation_bits, min_val=0, max_val=1)


    def forward(self, x):
        #out = self.quant_inp(x)

        out = self.relu1(self.conv1(x))
        out = self.max_pool1(out)

        out = self.relu2(self.conv2(out))
        out = self.max_pool2(out)

        out = self.relu3(self.conv3(out))
        out = self.max_pool3(out)

        out = self.relu4(self.conv4(out))
        out = self.max_pool4(out)

        out = self.relu5(self.conv5(out))
        out = self.max_pool5(out)

        out = self.relu6(self.conv6(out))
        out = self.max_pool6(out)

        out = self.relu7(self.conv7(out))
        out = self.relu8(self.conv8(out))
        out = self.relu9(self.conv9(out))

        out = self.hard_tanh(self.conv10(out))

        print(out.size())

        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)

        print(out.size())

        return out


