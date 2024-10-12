import torch
import torch.nn as nn

# from torch.nn import Module, Conv2d, Parameter, Softmax

from torch.nn import Module, Conv2d, Parameter
import numpy as np

from torch.autograd import Function

# class Softmax(Function):
#     @staticmethod
#     def forward(ctx, input):

#         ctx.save_for_backward(input)
#         # result = torch.softmax(input, dim=-1)
#         result = torch.nn.functional.softmax(input, dim=-1)

#         return result
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_variables
#         grad_input = grad_output.clone()
#         # grad_input = grad_input * input.softmax(dim=-1) * (1.0 - input.softmax(dim=-1))
#         grad_input = grad_input * torch.nn.functional.softmax(input, dim=-1) * (1.0 - torch.nn.functional.softmax(input, dim=-1))
        
#         return grad_input

'''
# -----------------------------------------------------------------------------------------------
# original PAM_Module
# -----------------------------------------------------------------------------------------------
class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        # clx ---------------------------------------------------------
        self.softmax = torch.nn.LeakyReLU()
        # -------------------------------------------------------------

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        # 这个 bmm 在转 rknn 时会转换，不影响
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
'''

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        # clx ---------------------------------------------------------
        self.leakyrelu = torch.nn.LeakyReLU()
        self.nonlinear = torch.nn.ELU()
        self.selu = torch.nn.SELU()
        self.sigmoid = torch.nn.Sigmoid()

        self.clx_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1)
        self.conv1d = torch.nn.Conv1d(in_channels=3600, out_channels=3600, kernel_size=1, stride=1)
        # -------------------------------------------------------------

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        # original ------------------------------------------------------------------------------
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # clx -----------------------------------------------------------------------------------
        # 经验证此修改仍可以训练出有效模型（从 rknn 结构可视化上，确实去掉了一个 Transpose）
        proj_query = self.query_conv(x).view(m_batchsize, -1, C//8)
        # ---------------------------------------------------------------------------------------

        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        # original code: 这个 bmm 在转 rknn 时会转换，不影响 ----------------------------------------
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # ---------------------------------------------------------------------------------------
        
        # clx: 尝试去掉 PAM 模块中第二个 Transpose --------------------------------------------------
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.leakyrelu(energy)
        # 【原始】：先对 attention 进行转置，再进行矩阵乘 
        # permuted_attention = attention.permute(0, 2, 1)
        # out = torch.bmm(proj_value, permuted_attention)
        
        # 【方法一】：attention.shape: 1*3600*3600, 转置(0, 2, 1) 仍旧为 1*3600*3600; 尝试不进行转置，直接进行矩阵乘
        # 从中间 ckpt 转 rknn 的结构可视化上看，可以再次去掉一个 Transpose,顺便还去掉了一个矩阵乘
        # 验证结果的正确性 --> 精度上台阶预测损失比较大
        # out = torch.bmm(proj_value, attention)

        # 【方法二】：因方法一不转置台阶预测损失比较大，因此尝试添加 Conv 替代 permute，看是否可以弥补损失的精度
        # 为了使用卷积，先将 attention 扩充一个维度，卷积操作结束后，再去掉这个维度
        # 该方法可以弥补台阶的一些损失，但是仍旧有明显的损失。且 Reshape 耗时明显增加，并没有影响 矩阵乘 的数量
        # attention = attention.view(m_batchsize, 1, width*height, width*height)
        # attention = self.clx_conv(attention)
        # attention = attention.view(m_batchsize, width*height, width*height)
        # out = torch.bmm(proj_value, attention)

        # 【方法三】：方法二增加了两个 view，在转 RKNN 时 Reshape 十分耗时，为了解决这个问题，不扩充到 4维，直接增加conv1d
        # 不仅去掉了第二个 Transpose 节点，同时去掉了一个 MatMul，且模型效果影响不大
        # 但是模型的参数量变得非常大,从 1.7M 增加到 51.2M
        attention = self.conv1d(attention)
        out = torch.bmm(proj_value, attention)
        # ---------------------------------------------------------------------------------------
        

        # clx ----------------------------------------------------------------
        # torch.matmul 等效于 torch.bmm :out = torch.matmul(proj_value, attention.permute(0, 2, 1))
        # 方法一 ：将 3维 的 torch.bmm 转换为 2 维的 torch.mm 实现
        # permuted_att = attention.permute(0, 2, 1).cuda()
        # out = torch.zeros((proj_value.shape[0], proj_value.shape[1], permuted_att.shape[2]))
        # for i in range(proj_value.shape[0]):
        #     out[i] = torch.mm(proj_value[i].cuda(), permuted_att[i].cuda())
        #     # out[i] = torch.mm(proj_value[i].half().cuda(), permuted_att[i].half().cuda())
        # out = out.half().cuda()
        # 
        # 方法二：在 attention.permute后增加一个非线性激活 ，转 rknn 时可以自动将矩阵乘法合并进 conv
        # TODO：模型正确性待验证(torch/onnx结果正确，rknn结果不正确)
        # permuted_attention = attention.permute(0, 2, 1)
        # clx_attention = self.softmax(permuted_attention)
        # out = torch.bmm(proj_value, clx_attention)
        # 
        # 方法三：使用 ELU 替换 方法二中的 LeakyReLU
        # permuted_attention = attention.permute(0, 2, 1)
        # clx_attention = self.nonlinear(permuted_attention)
        # out = torch.bmm(proj_value, clx_attention)
        # 
        # 方法四：使用 SELU 替换 方法二中的 LeakyReLU
        # permuted_attention = attention.permute(0, 2, 1)
        # clx_attention = self.selu(permuted_attention)
        # out = torch.bmm(proj_value, clx_attention)

        # 方法五：参考文档02 page 163 子图融合：Split + Sigmoid + Mul --> GLU, 在 bmm前增加 Sigmoid
        # permuted_attention = attention.permute(0, 2, 1)
        # clx_attention = self.sigmoid(permuted_attention)
        # out = torch.bmm(proj_value, clx_attention)

        # 方法六：在 permute 前加一个 conv，是否可以去掉一个 Transpose
        
        # --------------------------------------------------------------------

        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

# # ----------------------------------------------------------------------------------------
# # 使用 高效卷积运算，重新改写 PAM 模块
# # ----------------------------------------------------------------------------------------
# class PAM_Module(Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.gamma = Parameter(torch.zeros(1))
#         # clx ---------------------------------------------------------
#         self.leakyrelu = torch.nn.LeakyReLU()
#         self.clx_conv2d_extend = torch.nn.Conv2d(
#             in_channels = self.chanel_in, 
#             out_channels = 2*self.chanel_in, 
#             kernel_size = 1, 
#             stride = 1
#         )
#         self.clx_conv2d_squz = torch.nn.Conv2d(
#             in_channels = 2*self.chanel_in, 
#             out_channels = self.chanel_in, 
#             kernel_size = 1, 
#             stride=1
#         )
#         self.clx_conv2d = torch.nn.Conv2d(
#             in_channels = self.chanel_in, 
#             out_channels = self.chanel_in, 
#             kernel_size = 1, 
#             stride=1
#         )
#         self.bn_extend = nn.BatchNorm2d(2 * self.chanel_in, eps=1e-03)
#         self.act_extend = nn.PReLU(2 * self.chanel_in)
#         self.bn_squz = nn.BatchNorm2d(self.chanel_in, eps=1e-03)
#         self.act_squz = nn.PReLU(self.chanel_in)
#         self.bn = nn.BatchNorm2d(self.chanel_in, eps=1e-03)
#         self.act = nn.PReLU(self.chanel_in)
#         # -------------------------------------------------------------


#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         # m_batchsize, C, height, width = x.size()    # bs, 32, 45, 80 (input_shape: 360, 640)

#         # extend
#         out = self.clx_conv2d_extend(x)
#         out = self.bn_extend(out)
#         out = self.act_extend(out)
#         # squeeze
#         out = self.clx_conv2d_squz(out)
#         out = self.bn_squz(out)
#         out = self.act_squz(out)
#         # conv
#         out = self.clx_conv2d(out)
#         out = self.bn(out)
#         out = self.act(out)

#         out = self.gamma*out + x
#         return out




# ---------------------------------------------------------------------
# original  
# ---------------------------------------------------------------------
class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        # clx ---------------------------------------------------------
        self.leakyrelu = torch.nn.LeakyReLU()
        # -------------------------------------------------------------


    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()    # bs, 32, 45, 80 (input_shape: 360, 640)
        proj_query = x.view(m_batchsize, C, -1)     # bs, 32, 3600
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # bs, 3600, 32

        energy = torch.bmm(proj_query, proj_key)   # bs, 32, 32
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.leakyrelu(energy_new)  # bs, 32, 32
        proj_value = x.view(m_batchsize, C, -1) # bs, 32, 3600

        out = torch.bmm(attention, proj_value)  # bs, 32, 3600
        out = out.view(m_batchsize, C, height, width)   # bs, 32, 45, 80

        out = self.gamma*out + x
        return out


# # 使用高效的卷积，改写整个 CAM 模块
# class CAM_Module(Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim):
#         super(CAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.gamma = Parameter(torch.zeros(1))
#         # clx ---------------------------------------------------------
#         self.leakyrelu = torch.nn.LeakyReLU()
#         self.clx_conv2d_extend = torch.nn.Conv2d(
#             in_channels = self.chanel_in, 
#             out_channels = 2*self.chanel_in, 
#             kernel_size = 1, 
#             stride = 1
#         )
#         self.clx_conv2d_squz = torch.nn.Conv2d(
#             in_channels = 2*self.chanel_in, 
#             out_channels = self.chanel_in, 
#             kernel_size = 1, 
#             stride=1
#         )
#         self.bn_extend = nn.BatchNorm2d(2 * self.chanel_in, eps=1e-03)
#         self.act_extend = nn.PReLU(2 * self.chanel_in)
#         self.bn_squz = nn.BatchNorm2d(self.chanel_in, eps=1e-03)
#         self.act_squz = nn.PReLU(self.chanel_in)
#         # -------------------------------------------------------------


#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         # m_batchsize, C, height, width = x.size()    # bs, 32, 45, 80 (input_shape: 360, 640)

#         # extend
#         out = self.clx_conv2d_extend(x)
#         out = self.bn_extend(out)
#         out = self.act_extend(out)
#         # squeeze
#         out = self.clx_conv2d_squz(out)
#         out = self.bn_squz(out)
#         out = self.act_squz(out)

#         # proj_query = x.view(m_batchsize, C, -1)     # bs, 32, 3600
#         # proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # bs, 3600, 32

#         # energy = torch.bmm(proj_query, proj_key)   # bs, 32, 32
#         # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         # attention = self.leakyrelu(energy_new)  # bs, 32, 32
#         # proj_value = x.view(m_batchsize, C, -1) # bs, 32, 3600

#         # out = torch.bmm(attention, proj_value)  # bs, 32, 3600
#         # out = out.view(m_batchsize, C, height, width)   # bs, 32, 45, 80

#         out = self.gamma*out + x
#         return out


class UPx2(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    def fuseforward(self, input):
        output = self.deconv(input)
        output = self.act(output)
        return output

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output
    def fuseforward(self, input):
        output = self.conv(input)
        output = self.act(output)
        return output
    

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        # print(nIn, nOut, (kSize, kSize))
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        #combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output
class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut=nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        # print("bf bn :",input.size(),self.nOut)
        output = self.bn(input)
        # print("after bn :",output.size())
        output = self.act(output)
        # print("after act :",output.size())
        return output
class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        # print(nIn,n,n1,"--")
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        # print("nOut bf :",nOut)
        self.bn = BR(nOut)
        # print("nOut at :",self.bn.size())
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        # print(d1.size(),add1.size(),add2.size(),add3.size(),add4.size())

        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # print("combine :",combine.size())
        # if residual version
        if self.add:
            # print("add :",combine.size())
            combine = input + combine
        # print(combine.size(),"-----------------")
        output = self.bn(combine)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input

class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, p=5, q=3):
    # def __init__(self, classes=20, p=1, q=1):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = CBR(16 + 3,19,3)
        self.level2_0 = DownSamplerB(16 +3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = CBR(128 + 3,131,3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        # self.mixstyle = MixStyle2(p=0.5, alpha=0.1)
        self.b3 = CBR(256,32,3)
        self.sa = PAM_Module(32)
        self.sc = CAM_Module(32)
        self.conv_sa = CBR(32,32,3)
        self.conv_sc = CBR(32,32,3)
        self.classifier = CBR(32, 32, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        cat_=torch.cat([output2_0, output2], 1)

        output2_cat = self.b3(cat_)
        out_sa=self.sa(output2_cat)
        out_sa=self.conv_sa(out_sa)
        out_sc=self.sc(output2_cat)
        out_sc=self.conv_sc(out_sc)
        out_s=out_sa+out_sc
        classifier = self.classifier(out_s)

        return classifier

class TwinLiteNet(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, p=2, q=3, ):

        super().__init__()
        self.encoder = ESPNet_Encoder(p, q)

        self.up_1_1 = UPx2(32,16)
        self.up_2_1 = UPx2(16,8)

        self.up_1_2 = UPx2(32,16)
        self.up_2_2 = UPx2(16,8)

        self.classifier_1 = UPx2(8,2)
        self.classifier_2 = UPx2(8,2)



    def forward(self, input):

        x=self.encoder(input)
        x1=self.up_1_1(x)
        x1=self.up_2_1(x1)
        classifier1=self.classifier_1(x1)
        
        

        x2=self.up_1_2(x)
        x2=self.up_2_2(x2)
        classifier2=self.classifier_2(x2)

        return (classifier1,classifier2)


