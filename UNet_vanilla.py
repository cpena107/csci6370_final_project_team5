import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

class DoubleConv(nn.Module):
    """
    A double convolution block consisting of two consecutive Conv2d layers
    with ReLU activation, used as a basic building block throughout the network.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    """
    Downsampling block that performs a double convolution followed by max pooling.
    Returns both the convolved features and the pooled output for skip connections.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    """
    Upsampling block that performs transposed convolution followed by concatenation
    with skip connections and a double convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], dim=1)
       return self.conv(x)

class Encoder(nn.Module):
    """
    The encoder part of the UNet architecture, consisting of four downsampling blocks
    that progressively reduce spatial dimensions while increasing feature channels.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        return down_1, down_2, down_3, down_4, p4
    
class Decoder(nn.Module):
    """
    The decoder part of the UNet architecture, consisting of four upsampling blocks
    that progressively increase spatial dimensions while decreasing feature channels.
    """
    def __init__(self):
        super().__init__()
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
    def forward(self, x1, x2, x3, x4, x5):
        up_1 = self.up_convolution_1(x5, x4)
        up_2 = self.up_convolution_2(up_1, x3)
        up_3 = self.up_convolution_3(up_2, x2)
        up_4 = self.up_convolution_4(up_3, x1)
        return up_1, up_2, up_3, up_4

class SegmentationHead(nn.Module):
    """
    Final layer of the network that converts features to class predictions
    using a 1x1 convolution to match the number of output classes.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Complete UNet architecture for semantic segmentation, combining encoder,
    bottleneck, decoder, and segmentation head. Includes methods for loading
    pretrained weights from ResNet50 and DeepLabV3.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.encoder = Encoder(in_channels)

        self.bottle_neck = DoubleConv(512, 1024)

        self.decoder = Decoder()
        
        self.out = SegmentationHead(64, num_classes)

    def forward(self, x):
       down_1, down_2, down_3, down_4, p4 = self.encoder(x)

       b = self.bottle_neck(p4)

       up_1, up_2, up_3, up_4 = self.decoder(down_1, down_2, down_3, down_4, b)

       out = self.out(up_4)
       return out

    def load_from(self, model_path=None):
        if model_path is not None:
            print(f"pretrained_path:{model_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(model_path, map_location=device)
            self.load_state_dict(pretrained_dict)

    def load_deeplab_weights(self):
        deeplab =  deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, pretrained=True)
        self.encoder.load_state_dict(deeplab.state_dict(), strict=False)
        print("DeepLabV3 weights loaded for fine-tuning")

if __name__ == "__main__":
    double_conv = DoubleConv(256, 256)
    print(double_conv)

    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    output = model(input_image)
    print(output.size())