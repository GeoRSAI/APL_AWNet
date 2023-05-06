import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class SELR():
    def __init__(self, x):
        super(SELR, self).__init__()
        self.x = x

    def Tanh(self):
        y = np.tanh(self.x)*0.5+0.5
        y_grad = 0.5 - 0.5*y * y
        return [y, y_grad]


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class AWModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(AWModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
#一层
class AWNet(AWModule):
    def __init__(self, input, hidden1, output):
        super(AWNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)
        return out

# 两层
# class AWNet(AWModule):
#     def __init__(self, input, hidden1, hidden2, output):
#     # def __init__(self, input, hidden1, hidden2, hidden3, output):
#         super(AWNet, self).__init__()
#         self.linear1 = MetaLinear(input, hidden1)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.linear2 = MetaLinear(hidden1, hidden2)
#         # self.dropout = nn.Dropout(p=0.1)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.linear3 = MetaLinear(hidden2, output)
#         # self.linear4 = MetaLinear(hidden3, output)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         # x = self.dropout(x)
#         x = self.relu1(x)
#
#         # out = self.linear2(x)
#
#         x = self.linear2(x)
#         # x = self.sigmoid1(x)
#         # x = self.linear3(x)
#         x = self.relu2(x)
#         out = self.linear3(x)
#         return F.sigmoid(out)

#三层
# class AWNet(AWModule):
#     def __init__(self, input, hidden1, hidden2, hidden3, output):
#         super(AWNet, self).__init__()
#         self.linear1 = MetaLinear(input, hidden1)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.linear2 = MetaLinear(hidden1, hidden2)
#         # self.dropout = nn.Dropout(p=0.1)
#         self.sigmoid1 = nn.Sigmoid()
#         self.linear3 = MetaLinear(hidden2, hidden3)
#         self.linear4 = MetaLinear(hidden3, output)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         # x = self.dropout(x)
#         x = self.relu1(x)
#         x = self.linear2(x)
#         x = self.sigmoid1(x)
#         x = self.linear3(x)
#         x = self.sigmoid1(x)
#         out = self.linear4(x)
#         return out