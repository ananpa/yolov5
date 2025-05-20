# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Activation functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GSigmoidV1(nn.Module):
    """
    output = 1 / [1 + exp(-alpha * (x - beta))]
    """
    def __init__(self, alpha_init=1.0, beta_init=0.0):
        super().__init__()
        self.alpha = alpha_init
        self.beta = beta_init

    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-self.alpha * (x - self.beta)))
    
    def __repr__(self):
        return f'GSigmoidV1(alpha={self.alpha}, beta={self.beta})'
    
class GeneralizedSigmoid(nn.Module):
    """
    output = 1 / [1 + exp(-alpha * (x - beta))]
    where alpha, beta are learnable parameters.
    """
    def __init__(self, alpha_init=1.0, beta_init=0.0):
        super().__init__()
        # Make alpha, beta trainable (learnable) parameters
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-self.alpha * (x - self.beta)))

    def __repr__(self):
        return f'GeneralizedSigmoid(a={self.alpha}, b={self.beta})'
        
class PELU(nn.Module):
    def __init__(self, a_init=1.0, b_init=1.0):
        super().__init__()
        #Learnable parameters:
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float))

    def forward(self, x):
        # PELU is piecewise-defined, so we mask x >= 0 vs x < 0
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        x_pos = x[pos_mask]
        x_neg = x[neg_mask]

        # PELU piecewise
        # for x >= 0:  (a/b) * x
        # for x < 0:   a * (exp(x/b) - 1)
        y_pos = (self.a / self.b) * x_pos
        y_neg = self.a * (torch.exp(x_neg / self.b) - 1)

        # Recombine
        out = torch.zeros_like(x)
        out[pos_mask] = y_pos
        out[neg_mask] = y_neg
        return out

# Hybrid Activation Unit (ReLU + SiLU)
class ReLUSiLU(nn.Module):
    def __init__(self, init_alpha=0.5):
        super(ReLUSiLU, self).__init__()
        # Alpha parameter balances the two activations
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        relu_out = F.relu(x)
        silu_out = F.silu(x)  # Swish activation
        # Combine activations adaptively
        return self.alpha * relu_out + (1 - self.alpha) * silu_out

class ReLUELU(nn.Module):
    def __init__(self, alpha_init=1.0):
        super(ReLUELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))  # Trainable alpha parameter

    def forward(self, x):
        return torch.where(x > 0, self.alpha * x, self.alpha * (torch.exp(x) - 1))
    
class Smish(nn.Module):
    """Applies the Smish activation function, a smooth approximation of ReLU."""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    #@staticmethod
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))
    
class SiLU(nn.Module):
    """Applies the Sigmoid-weighted Linear Unit (SiLU) activation function, also known as Swish."""

    @staticmethod
    def forward(x):
        """
        Applies the Sigmoid-weighted Linear Unit (SiLU) activation function.

        https://arxiv.org/pdf/1606.08415.pdf.
        """
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Applies the Hardswish activation function, which is efficient for mobile and embedded devices."""

    @staticmethod
    def forward(x):
        """
        Applies the Hardswish activation function, compatible with TorchScript, CoreML, and ONNX.

        Equivalent to x * F.hardsigmoid(x)
        """
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX


class Mish(nn.Module):
    """Mish activation https://github.com/digantamisra98/Mish."""

    @staticmethod
    def forward(x):
        """Applies the Mish activation function, a smooth alternative to ReLU."""
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    """Efficiently applies the Mish activation function using custom autograd for reduced memory usage."""

    class F(torch.autograd.Function):
        """Implements a custom autograd function for memory-efficient Mish activation."""

        @staticmethod
        def forward(ctx, x):
            """Applies the Mish activation function, a smooth ReLU alternative, to the input tensor `x`."""
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            """Computes the gradient of the Mish activation function with respect to input `x`."""
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        """Applies the Mish activation function to the input tensor `x`."""
        return self.F.apply(x)


class FReLU(nn.Module):
    """FReLU activation https://arxiv.org/abs/2007.11824."""

    def __init__(self, c1, k=3):  # ch_in, kernel
        """Initializes FReLU activation with channel `c1` and kernel size `k`."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        """
        Applies FReLU activation with max operation between input and BN-convolved input.

        https://arxiv.org/abs/2007.11824
        """
        return torch.max(x, self.bn(self.conv(x)))


class AconC(nn.Module):
    """
    ACON activation (activate or not) function.

    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    See "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
    """

    def __init__(self, c1):
        """Initializes AconC with learnable parameters p1, p2, and beta for channel-wise activation control."""
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        """Applies AconC activation function with learnable parameters for channel-wise control on input tensor x."""
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    """
    ACON activation (activate or not) function.

    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    See "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
    """

    def __init__(self, c1, k=1, s=1, r=16):
        """Initializes MetaAconC with params: channel_in (c1), kernel size (k=1), stride (s=1), reduction (r=16)."""
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        """Applies a forward pass transforming input `x` using learnable parameters and sigmoid activation."""
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x
