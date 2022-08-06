# Credit goes to https://github.com/drorsimon/CSCNet
# Simon, Dror, and Michael Elad. "Rethinking the CSC model for natural images." Advances in Neural Information Processing Systems 32 (2019).

from collections import namedtuple


import torch
import torch.nn as nn
import torch.nn.functional as F


ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings', 'channels'])
def calc_pad_sizes(I: torch.Tensor, kernel_size: int, stride: int):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 else stride - ((I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 else stride - ((I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad

class SoftThreshold(nn.Module):
    def __init__(self, size, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out





class ConvLista_T(nn.Module):
    def __init__(self, params: ListaParams, A=None, B=None, C=None, threshold=1e-2, norm=False):
        super(ConvLista_T, self).__init__()
        if A is None:
            A = torch.randn(params.num_filters, params.channels, params.kernel_size, params.kernel_size)
            l = conv_power_method(A, [128, 128], num_iters=20, stride=params.stride)
            # l = conv_power_method(A, [28,28], num_iters=200, stride=params.stride)
            A /= torch.sqrt(l)
        if B is None:
            B = torch.clone(A)
        if C is None:
            C = torch.clone(A)
        self.apply_A = torch.nn.ConvTranspose2d(params.num_filters, params.channels, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_B = torch.nn.Conv2d(params.channels, params.num_filters, kernel_size=params.kernel_size, stride=params.stride, bias=False)
        self.apply_C = torch.nn.ConvTranspose2d(params.num_filters, params.channels, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_A.weight.data = A
        self.apply_B.weight.data = B
        self.apply_C.weight.data = C
        self.soft_threshold = SoftThreshold(params.num_filters, threshold)
        self.params = params
        self.num_iter = params.unfoldings
        # self.norm = norm
        # if self.norm:
            # self.norm_layer = torch.nn.InstanceNorm2d(params.num_filters)
            # self.norm_layer = torch.nn.

    def _split_image(self, I):
        if self.params.stride == 1:
            return I, torch.ones_like(I)
        left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(I, self.params.kernel_size, self.params.stride)
        I_batched_padded = torch.zeros(I.shape[0], self.params.stride ** 2, I.shape[1], top_pad + I.shape[2] + bot_pad,
                                       left_pad + I.shape[3] + right_pad).type_as(I)
        valids_batched = torch.zeros_like(I_batched_padded)
        for num, (row_shift, col_shift) in enumerate([(i, j) for i in range(self.params.stride) for j in range(self.params.stride)]):
            I_padded = F.pad(I, pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
            valids = F.pad(torch.ones_like(I), pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='constant')
            I_batched_padded[:, num, :, :, :] = I_padded
            valids_batched[:, num, :, :, :] = valids
        I_batched_padded = I_batched_padded.reshape(-1, *I_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return I_batched_padded, valids_batched
    
    def disable_warmup(self):
        self.num_iter = self.params.unfoldings
        
    def enable_warmup(self):
        self.num_iter = 1
    
    def forward(self, I):
        I_batched_padded, valids_batched = self._split_image(I)
        conv_input = self.apply_B(I_batched_padded) #encode
        gamma_k = self.soft_threshold(conv_input)
        # ic(gamma_k.shape)
        for k in range(self.num_iter - 1):
            x_k = self.apply_A(gamma_k) # decode
            # r_k = self.apply_B(x_k-I_batched_padded) #encode
            r_k = self.apply_B(x_k-I_batched_padded) #encode
            # if self.norm:
                # r_k = self.norm_layer(r_k)
                #bug? try adding
            gamma_k = self.soft_threshold(gamma_k - r_k)
        output_all = self.apply_C(gamma_k)
        output_cropped = torch.masked_select(output_all, valids_batched.bool()).reshape(I.shape[0], self.params.stride ** 2, *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output = output_cropped.mean(dim=1, keepdim=False)
        # output = F.relu(output)
        return torch.clamp(output,0.0,1.0)
    
    

def conv_power_method(D, image_size, num_iters=100, stride=1):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    needles_shape = [int(((image_size[0] - D.shape[-2])/stride)+1), int(((image_size[1] - D.shape[-1])/stride)+1)]
    x = torch.randn(1, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = F.conv_transpose2d(x, D, stride=stride)
        x = F.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))




