import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    FairseqDropout,
    LayerNorm)


from numpy.linalg import svd

class QuantizeLinear(nn.Module):
    def __init__(self, weight, bias, out_features, g=None, h=None, 
                #  bias=True
                 ):
        super(QuantizeLinear, self).__init__()
        in_features, out_features = weight.size()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())
        
        self.numpy_weight = weight.numpy()
        
        U, S, VT = svd(self.numpy_weight)
        

        
        U = torch.from_numpy(U)
        VT = torch.from_numpy(VT)
        
# 秩-1逼近向量
        U_rank_1_approximation = U[:, 0]
        Vt_rank_1_approximation = VT[0, :]

        


        if g is None:
            self.g = nn.Parameter(Vt_rank_1_approximation) 
        else:
            self.g = nn.Parameter(g.clone())
        if h is None:
            self.h = nn.Parameter(U_rank_1_approximation)  # in_features
        else:
            self.h = nn.Parameter(h.clone())
        # if bias:
        #     
        # else:
        #     self.register_parameter('bias', None)
        
        
        self.layer_norm = LayerNorm(in_features)

    def forward(self, input):
        # 计算Sign(W)
        sign_w = torch.sign(self.weight)

        # 计算X ⊙ g(WT±1)
        # g_norm = torch.norm(self.g, dim=0)
        # g_normalized = self.g / g_norm
        # x_g = input * torch.matmul(g_normalized, sign_w.t())
        x_g = input * self.g
        
        # x_g = x_g @ sign_w
        # x_g = torch.matmul(x_g, sign_w)
        x_g = F.linear(x_g, sign_w, self.bias)
        
        
        y = x_g * self.h
        
        # 计算Z = LayerNorm(Y)
        # z = nn.LayerNorm(y.size()[1:]).to(y.device)
        output = self.layer_norm(y)
        

        
        return output
    


def check_string(name, string_list):
    for item in string_list:
        if name in item:
            
            return True
    
    return False

def replace_linear_layers_with_quantize_recursive(module):
    check_string_list = ["output_projection", "post_extract_proj"]
    for name, child_module in module.named_children():

        if isinstance(child_module, nn.Linear) and not check_string(name,check_string_list ) :

        # if isinstance(child_module, nn.Linear) and ("output_projection" not in name):
            # 保存原始权重参数
            weight = child_module.weight.data
            
            bias = child_module.bias.data
            
            quantize_linear = QuantizeLinear(weight, bias, child_module.out_features)
            setattr(module, name, quantize_linear)
        else:
            # 递归调用，继续遍历子模块
            replace_linear_layers_with_quantize_recursive(child_module)


# 示例用法


