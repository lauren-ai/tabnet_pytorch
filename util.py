# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/21 15:23
@Auth ： zhlhong
@File ：utils.py
@IDE ：PyCharm
@Email：zhlhong@szlanyou.com
"""

from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import torch

"""
Other possible implementations:
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
https://github.com/msobroza/SparsemaxPytorch/blob/master/mnist/sparsemax.py
https://github.com/vene/sparse-structured-attention/blob/master/pytorch/torchsparseattn/sparsemax.py
"""


# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(input, dim=0):
    d = input.size(dim)  # 每个样本的长度
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)   #
    view = [1] * input.dim()  # 输入向量的维度，并设置为1
    view[0] = -1    # 第0维设置为-1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    添加自定义算子
    1. 获取原算子的前向推理接口。
    2. 获取目标 ONNX 算子的定义。
    3. 编写符号函数并绑定。
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)   # 去除每个样本中的最大值和索引
        input -= max_val  # same numerical stability trick as for softmax  每个样本都减去最大值
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)   # 为每个样本筛选出一个满足条件的那个
        output = torch.clamp(input - tau, min=0)    # 对tensor按照指定范围进行裁剪,并将小于0的位置取为min的值0
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):   # loss.backward() 处更新
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)  # 对每个样本都减去最大值的结果进行排序
        input_cumsum = input_srt.cumsum(dim) - 1        # 对张量中每个样本的值不断求累计和
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)  # [3,1]
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size

    # @staticmethod
    # def symbolic(g, input):   # torch._C.Value
        # return g.op("onnx_test::sparsemax_", *[input], **{})
        # return g.op("onnx_test::sparsemax_", input)
        # return g.op("custom::sparsemax", input, g.op("Constant", value_t=torch.tensor([], dtype=torch.int)), dim,
        #             coordinate_transformation_mode_s="pytorch_half_pixel", cubic_coeff_a_f=-0.75, mode_s='cubic',
        #             nearest_mode_s="floor")



# def sparsemax(input):
#     return SparsemaxFunction.apply(input)

class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def symbolic(g, input: torch._C.Value) -> torch._C.Value:    # torch._C.graph
        return g.op("Clip", input, g.op("Constant", value_t=torch.tensor(0, dtype=torch.float)))

relu5 = MyRelu.apply

sparsemax = SparsemaxFunction.apply

from torch.onnx import register_custom_op_symbolic
# register_custom_op_symbolic('sparsemax',SparsemaxFunction, symbolic_fn=None)
register_custom_op_symbolic('::sparsemax',symbolic_fn=SparsemaxFunction,opset_version=9)

class Sparsemax(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()
        # self.relu = nn.ReLU()

    def forward(self, input):
        # return sparsemax(input, self.dim)
        return sparsemax(input)
        # return relu5(input)




class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(Function):
    """ A highly optimized equivalent of lambda x: Entmax15([x, 0]) """

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply


class Entmax15(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)


if __name__ == '__main__':
    # 测试Sparsemax 算子是否转成功onnx算子
    model = Sparsemax(dim=-1)

    input = torch.rand(1, 3)
    model.eval()
    torch.onnx.export(model, input, 'model_revised.onnx')
    torch_output = model(input).detach().numpy()

    import onnxruntime
    import numpy as np
    import onnx

    sess = onnxruntime.InferenceSession('model_revised.onnx')
    in_namea = sess.get_inputs()[0].name
    # in_nameb = sess.get_inputs()[1].name
    out_name = sess.get_outputs()[0].name

    # ort_output = sess.run(None, {in_namea: input.numpy()})[0]
    ort_output = sess.run([out_name], {in_namea: input.numpy()})

    assert np.allclose(torch_output, ort_output)
