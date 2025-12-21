"""
ELBO损失函数实现

L_ELBO = -E_q[log p(Y|X,W)] + KL[q(W)||p(W)]
       = L_rec + λ * L_KL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing交叉熵损失
    
    Args:
        epsilon: 平滑参数
        weight: 类别权重
        reduction: 归约方式
    """
    def __init__(self, epsilon=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits, target):
        """
        Args:
            logits: 模型输出 (batch, num_classes)
            target: 真实标签 (batch,)
        
        Returns:
            损失值
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 构建平滑标签分布
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        
        # 计算损失
        loss_per_sample = -torch.sum(true_dist * log_probs, dim=-1)
        
        # 应用类别权重
        if self.weight is not None:
            sample_weights = self.weight[target]
            loss_per_sample = loss_per_sample * sample_weights
        
        # 归约
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class ELBOLoss(nn.Module):
    """
    ELBO损失函数
    
    L_ELBO = L_rec + λ * L_KL
    
    Args:
        model: BSTGCN模型
        kl_weight: KL散度权重
        label_smoothing: Label Smoothing参数
        class_weights: 类别权重
        num_batches: 训练集batch数量 (用于归一化KL)
    """
    def __init__(
        self,
        model,
        kl_weight=1.0,
        label_smoothing=0.05,
        class_weights=None,
        num_batches=1
    ):
        super().__init__()
        
        self.model = model
        self.kl_weight = kl_weight
        self.num_batches = num_batches
        
        # 重建损失 (保留Label Smoothing)
        self.ce_loss = LabelSmoothingCrossEntropy(
            epsilon=label_smoothing,
            weight=class_weights
        )
    
    def forward(self, logits, targets):
        """
        计算ELBO损失

        Args:
            logits: 模型输出 (batch, num_classes)
            targets: 真实标签 (batch,)

        Returns:
            total_loss: 总损失
            rec_loss: 重建损失
            kl_loss: KL散度 (已归一化)
        """
        # 重建损失 (交叉熵)
        rec_loss = self.ce_loss(logits, targets)

        # KL散度 (收集所有贝叶斯层)
        kl_loss_raw = self.model.kl_divergence()

        # 归一化KL (按batch数和数据集大小)
        # 这样KL散度的量级与重建损失相当
        kl_loss = kl_loss_raw / self.num_batches

        # 总损失
        total_loss = rec_loss + self.kl_weight * kl_loss

        return total_loss, rec_loss, kl_loss


class HeteroscedasticELBOLoss(nn.Module):
    """
    Heteroscedastic ELBO损失函数

    用于分类任务的数据不确定性建模:
    - 使用学习到的方差对logits进行扰动
    - 通过MC采样近似期望
    - 添加方差正则化防止方差坍塌到0

    理论基础:
    - 将分类logits建模为高斯分布: f(x) ~ N(mu(x), sigma^2(x))
    - 预测概率: p(y|x) = E[softmax(f)]，通过MC采样近似
    - 数据不确定性由学习到的sigma(x)捕获

    Args:
        model: BSTGCN模型
        kl_weight: KL散度权重
        label_smoothing: Label Smoothing参数
        class_weights: 类别权重
        num_batches: 训练集batch数量
        num_mc_samples: 用于近似期望的MC采样次数
        min_variance: 最小方差 (数值稳定性)
        max_variance: 最大方差 (防止过大方差)
        variance_reg_weight: 方差正则化权重 (鼓励学习非零方差)
        target_variance: 目标方差 (正则化的目标值)
    """
    def __init__(
        self,
        model,
        kl_weight=1.0,
        label_smoothing=0.05,
        class_weights=None,
        num_batches=1,
        num_mc_samples=10,
        min_variance=1e-4,
        max_variance=10.0,
        variance_reg_weight=0.01,
        target_variance=0.5
    ):
        super().__init__()

        self.model = model
        self.kl_weight = kl_weight
        self.num_batches = num_batches
        self.num_mc_samples = num_mc_samples
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.variance_reg_weight = variance_reg_weight
        self.target_variance = target_variance

        # 标准交叉熵 (用于计算每个MC样本的损失)
        self.ce_loss = LabelSmoothingCrossEntropy(
            epsilon=label_smoothing,
            weight=class_weights,
            reduction='none'  # 返回每个样本的损失，用于MC平均
        )

    def forward(self, logits, log_variance, targets):
        """
        计算Heteroscedastic ELBO损失

        Args:
            logits: 模型输出的均值 (batch, num_classes)
            log_variance: 模型输出的log方差 (batch, num_classes)
            targets: 真实标签 (batch,)

        Returns:
            total_loss: 总损失
            rec_loss: 重建损失 (MC近似的交叉熵)
            kl_loss: KL散度 (已归一化)
        """
        # 将log_variance转换为方差，并限制范围
        variance = torch.clamp(
            F.softplus(log_variance),  # softplus保证正值
            min=self.min_variance,
            max=self.max_variance
        )
        std = torch.sqrt(variance)  # (B, C)

        # MC采样近似 E[CE(softmax(f + epsilon * sigma), y)]
        batch_size = logits.size(0)
        total_ce = torch.zeros(batch_size, device=logits.device)

        for _ in range(self.num_mc_samples):
            # 从标准正态分布采样
            epsilon = torch.randn_like(logits)
            # 对logits添加噪声: f' = mu + sigma * epsilon
            noisy_logits = logits + std * epsilon
            # 计算交叉熵
            ce = self.ce_loss(noisy_logits, targets)  # (B,)
            total_ce = total_ce + ce

        # 平均得到期望损失
        rec_loss = (total_ce / self.num_mc_samples).mean()

        # KL散度
        kl_loss_raw = self.model.kl_divergence()
        kl_loss = kl_loss_raw / self.num_batches

        # 方差正则化: 鼓励学习非零方差，防止方差坍塌
        # 使用log-space正则化: (log(variance) - log(target))^2
        # 这样对小方差和大方差都有惩罚
        log_var = torch.log(variance + 1e-8)
        log_target = torch.log(torch.tensor(self.target_variance, device=variance.device))
        variance_reg = ((log_var - log_target) ** 2).mean()

        # 总损失
        total_loss = rec_loss + self.kl_weight * kl_loss + self.variance_reg_weight * variance_reg

        return total_loss, rec_loss, kl_loss


class KLWeightScheduler:
    """
    KL权重退火调度器

    在训练初期逐渐增加KL权重，避免训练不稳定

    Args:
        kl_weight_max: 最大KL权重
        warmup_epochs: 预热轮数
        mode: 退火模式 ('linear' or 'exponential')
    """
    def __init__(self, kl_weight_max=1.0, warmup_epochs=10, mode='linear'):
        self.kl_weight_max = kl_weight_max
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.current_epoch = 0

    def step(self):
        """更新epoch"""
        self.current_epoch += 1

    def get_kl_weight(self):
        """获取当前KL权重"""
        if self.current_epoch >= self.warmup_epochs:
            return self.kl_weight_max

        if self.mode == 'linear':
            return self.kl_weight_max * (self.current_epoch / self.warmup_epochs)
        elif self.mode == 'exponential':
            return self.kl_weight_max * (1 - 0.5 ** (self.current_epoch / self.warmup_epochs))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

