"""
贝叶斯基础层实现

包含:
- BayesianLinear: 贝叶斯线性层
- BayesianConv1d: 贝叶斯1D卷积层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianLinear(nn.Module):
    """
    贝叶斯线性层
    
    使用变分推断学习权重分布 q(W; μ, ρ)
    采用重参数化技巧进行采样: W = μ + σ ⊙ ε, where ε ~ N(0,I)
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        bias: 是否使用偏置
        prior_std: 先验分布标准差
    """
    def __init__(self, in_features, out_features, bias=True, prior_std=1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.prior_std = prior_std
        
        # 权重的变分参数 (μ, ρ)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        
        # 偏置的变分参数
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化变分参数"""
        # μ: 使用Kaiming初始化
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))

        # ρ: 初始化为-7，使得初始σ非常小 (σ ≈ 0.001)
        # 这样初始KL散度会非常小，接近确定性模型
        nn.init.constant_(self.weight_rho, -7.0)

        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -7.0)
    
    def forward(self, x, sample=True):
        """
        前向传播
        
        Args:
            x: 输入 (*, in_features)
            sample: 是否采样 (训练时True，MAP推理时False)
        
        Returns:
            输出 (*, out_features)
        """
        if sample:
            # 计算标准差 (softplus保证正性)
            weight_sigma = F.softplus(self.weight_rho)
            
            # 重参数化采样
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_eps
            
            if self.use_bias:
                bias_sigma = F.softplus(self.bias_rho)
                bias_eps = torch.randn_like(self.bias_mu)
                bias = self.bias_mu + bias_sigma * bias_eps
            else:
                bias = None
        else:
            # MAP估计 (仅使用均值)
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        计算KL散度 KL[q(W)||p(W)]

        假设先验 p(W) = N(0, prior_std²)
        后验 q(W) = N(μ, σ²)

        KL = 0.5 * Σ[(σ/σ_prior)² + (μ/σ_prior)² - 1 - 2*log(σ/σ_prior)]

        使用稳定的计算方式避免数值溢出
        """
        # 权重KL
        weight_sigma = F.softplus(self.weight_rho)
        # 裁剪sigma避免数值不稳定
        weight_sigma = torch.clamp(weight_sigma, min=1e-6, max=5.0)

        # 裁剪mu避免极端值导致梯度爆炸
        weight_mu_clipped = torch.clamp(self.weight_mu, min=-10.0, max=10.0)

        # 使用更稳定的log计算
        log_sigma_ratio = torch.log(weight_sigma + 1e-8) - math.log(self.prior_std)

        kl_weight = 0.5 * torch.sum(
            (weight_sigma / self.prior_std) ** 2 +
            (weight_mu_clipped / self.prior_std) ** 2 -
            1 - 2 * log_sigma_ratio
        )

        # 裁剪KL避免极端值
        kl_weight = torch.clamp(kl_weight, min=0.0, max=1e6)

        # 偏置KL
        if self.use_bias:
            bias_sigma = F.softplus(self.bias_rho)
            bias_sigma = torch.clamp(bias_sigma, min=1e-6, max=5.0)
            bias_mu_clipped = torch.clamp(self.bias_mu, min=-10.0, max=10.0)

            log_bias_sigma_ratio = torch.log(bias_sigma + 1e-8) - math.log(self.prior_std)

            kl_bias = 0.5 * torch.sum(
                (bias_sigma / self.prior_std) ** 2 +
                (bias_mu_clipped / self.prior_std) ** 2 -
                1 - 2 * log_bias_sigma_ratio
            )
            kl_bias = torch.clamp(kl_bias, min=0.0, max=1e6)
        else:
            kl_bias = 0.0

        return kl_weight + kl_bias
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.use_bias}, prior_std={self.prior_std}'


class BayesianConv1d(nn.Module):
    """
    贝叶斯1D卷积层
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组卷积数
        bias: 是否使用偏置
        prior_std: 先验分布标准差
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, prior_std=1.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.prior_std = prior_std

        # 权重的变分参数
        self.weight_mu = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )

        # 偏置的变分参数
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_channels))
            self.bias_rho = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化变分参数"""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -7.0)

        if self.use_bias:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -7.0)

    def forward(self, x, sample=True):
        """
        前向传播

        Args:
            x: 输入 (batch, in_channels, length)
            sample: 是否采样权重

        Returns:
            输出 (batch, out_channels, length_out)
        """
        if sample:
            weight_sigma = F.softplus(self.weight_rho)
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_eps

            if self.use_bias:
                bias_sigma = F.softplus(self.bias_rho)
                bias_eps = torch.randn_like(self.bias_mu)
                bias = self.bias_mu + bias_sigma * bias_eps
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv1d(x, weight, bias, self.stride,
                       self.padding, self.dilation, self.groups)

    def kl_divergence(self):
        """计算KL散度 (使用稳定计算)"""
        weight_sigma = F.softplus(self.weight_rho)
        weight_sigma = torch.clamp(weight_sigma, min=1e-6, max=5.0)
        weight_mu_clipped = torch.clamp(self.weight_mu, min=-10.0, max=10.0)

        log_sigma_ratio = torch.log(weight_sigma + 1e-8) - math.log(self.prior_std)

        kl_weight = 0.5 * torch.sum(
            (weight_sigma / self.prior_std) ** 2 +
            (weight_mu_clipped / self.prior_std) ** 2 -
            1 - 2 * log_sigma_ratio
        )
        kl_weight = torch.clamp(kl_weight, min=0.0, max=1e6)

        if self.use_bias:
            bias_sigma = F.softplus(self.bias_rho)
            bias_sigma = torch.clamp(bias_sigma, min=1e-6, max=5.0)
            bias_mu_clipped = torch.clamp(self.bias_mu, min=-10.0, max=10.0)

            log_bias_sigma_ratio = torch.log(bias_sigma + 1e-8) - math.log(self.prior_std)

            kl_bias = 0.5 * torch.sum(
                (bias_sigma / self.prior_std) ** 2 +
                (bias_mu_clipped / self.prior_std) ** 2 -
                1 - 2 * log_bias_sigma_ratio
            )
            kl_bias = torch.clamp(kl_bias, min=0.0, max=1e6)
        else:
            kl_bias = 0.0

        return kl_weight + kl_bias

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'kernel_size={self.kernel_size}, stride={self.stride}, ' \
               f'padding={self.padding}, dilation={self.dilation}, ' \
               f'groups={self.groups}, bias={self.use_bias}, prior_std={self.prior_std}'

