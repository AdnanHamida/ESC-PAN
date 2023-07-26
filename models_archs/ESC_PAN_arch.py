from torch import nn,add,mul,cat
import math

class DSC(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
    super(DSC, self).__init__()
    self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=kernel_size//2, padding_mode='replicate')
    self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
  def forward(self, x):
    out = self.depthwise(x)
    out = self.pointwise(out)
    return out

class PA(nn.Module):
    """
    PA is pixel attention
    Taken from Github repo @ https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/archs/PAN_arch.py
    """
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = mul(x, y)

        return out
class ESCPA(nn.Module):
    """
    ESCPA is modified from SCPA (Zhao et al. "Efficient Image Super-Resolution Using Pixel Attention", ECCV 2020)
    Github: https://github.com/zhaohengyuan1/PAN
    """
    def __init__(self, C):
        super(ESCPA, self).__init__()

        self.upper_branch = nn.Sequential(
        nn.Conv2d(C, C//2, kernel_size=1, bias=False),
        nn.PReLU(),
        )

        self.K2 = nn.Sequential(
        nn.Conv2d(C//2, C//2, 1),
        nn.Sigmoid()
        )

        self.K3 = DSC(C//2, C//2, kernel_size=3, padding=1, bias=False)

        self.K4 = nn.Sequential(
        DSC(C//2, C//2, kernel_size=3, padding=1, bias=False),
        nn.PReLU()
        )

        self.lower_branch = nn.Sequential(
        nn.Conv2d(C, C//2, kernel_size=1, bias=False),
        nn.PReLU(),
        DSC(C//2, C//2, kernel_size=3, padding=1, bias=False), # K1
        nn.PReLU()
        )

        self.conv3 = nn.Conv2d(C, C, kernel_size=1, bias=False)

    def forward(self, X):

        #Upper branch
        X_dash = self.upper_branch(X)
        X_tilde = self.K2(X_dash)
        X_dash = self.K3(X_dash)
        X_dash =mul(X_dash, X_tilde)

        Y_dash = self.K4(X_dash)

        #Lower branch
        Y_dashdash = self.lower_branch(X)

        out = self.conv3(cat([Y_dash, Y_dashdash], dim=1))
        Y = out + X

        return Y

class ESC_PAN(nn.Module):
    def __init__(self, r, C =32, d = 1):
        super(ESC_PAN, self).__init__()
        self.scale_factor = r
        self.d = d
        self.C = C


        self.feature_extraction_module = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=5, stride=1, padding=2,padding_mode='replicate'),
            nn.PReLU(),
        )
        self.feature_mapping_module = nn.ModuleList([ESCPA(C = C) for i in range(d)])

        self.upsampling_module = nn.Sequential(
            DSC(in_channels= C, out_channels= C, kernel_size=3, padding=1),
            nn.PReLU(),
            PA(C),
            DSC(in_channels= C, out_channels= 1 * (r ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(r),
        )
    def forward(self, inputs):
        sc = nn.functional.interpolate(input=inputs, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        out = self.feature_extraction_module(inputs)
        for sc_conv in self.feature_mapping_module:
          out = (sc_conv(out))
        out = self.upsampling_module(out)
        out += sc
        return out
