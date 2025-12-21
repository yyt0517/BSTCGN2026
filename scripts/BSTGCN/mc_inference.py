"""
蒙特卡洛推理实现

包含:
- mc_inference: 标准MC推理
- mc_inference_heteroscedastic: Heteroscedastic模型的MC推理 (分解epistemic和aleatoric)
- adaptive_mc_inference: 自适应MC推理
- batch_mc_inference: 批量MC推理
"""

import torch
import torch.nn.functional as F


def mc_inference(model, x, num_samples=30, return_all=False, use_stable_entropy=True,
                 return_total_uncertainty=False, enable_dropout=False):
    """
    蒙特卡洛推理 (标准模式)

    Args:
        model: BSTGCN模型
        x: 输入数据 (batch, seq_len, features)
        num_samples: MC采样次数
        return_all: 是否返回所有采样结果
        use_stable_entropy: 是否使用数值稳定的熵计算
        return_total_uncertainty: 是否同时返回total uncertainty (预测熵)
        enable_dropout: 是否在推理时启用dropout (MC Dropout风格)

    Returns:
        pred_mean: 平均预测概率 (batch, num_classes)
        pred_class: 预测类别 (batch,)
        epistemic_uncertainty: 认知不确定性 (batch,)
        (可选) total_uncertainty: 总不确定性/预测熵 (batch,)
        (可选) all_probs: 所有采样的概率 (num_samples, batch, num_classes)
    """
    model.eval()

    # 如果启用dropout，在eval模式下保持dropout激活
    if enable_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    probs_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            # 每次采样不同的权重
            logits = model(x, sample=True)  # 内部自动采样
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

    # 聚合结果
    probs_stack = torch.stack(probs_list, dim=0)  # (M, B, C)
    pred_mean = probs_stack.mean(dim=0)  # (B, C)
    pred_class = pred_mean.argmax(dim=-1)  # (B,)

    # 计算不确定性
    # Total Uncertainty: H(E[p(Y|X,W)]) - 预测熵
    # Epistemic Uncertainty: I(Y;W|X) = H(E[p(Y|X,W)]) - E[H(p(Y|X,W))]
    if use_stable_entropy:
        # 使用PyTorch的稳定熵计算
        entropy_mean = torch.special.entr(pred_mean).sum(dim=-1)  # (B,) - Total Uncertainty
        entropy_samples = torch.special.entr(probs_stack).sum(dim=-1)  # (M, B)
    else:
        # 原始方法
        entropy_mean = -torch.sum(
            pred_mean * torch.log(pred_mean + 1e-10), dim=-1
        )  # (B,)

        entropy_samples = -torch.sum(
            probs_stack * torch.log(probs_stack + 1e-10), dim=-1
        )  # (M, B)

    mean_entropy = entropy_samples.mean(dim=0)  # (B,)
    total_uncertainty = entropy_mean  # (B,) - 预测熵
    epistemic_uncertainty = entropy_mean - mean_entropy  # (B,) - 互信息

    if return_all:
        if return_total_uncertainty:
            return pred_mean, pred_class, epistemic_uncertainty, total_uncertainty, probs_stack
        return pred_mean, pred_class, epistemic_uncertainty, probs_stack
    else:
        if return_total_uncertainty:
            return pred_mean, pred_class, epistemic_uncertainty, total_uncertainty
        return pred_mean, pred_class, epistemic_uncertainty


def mc_inference_heteroscedastic(
    model, x, num_samples=30, num_aleatoric_samples=10,
    use_stable_entropy=True, return_decomposition=True
):
    """
    Heteroscedastic模型的蒙特卡洛推理

    正确分解epistemic和aleatoric不确定性:
    - Epistemic (模型不确定性): 来自权重采样的预测差异
    - Aleatoric (数据不确定性): 来自学习到的方差

    分解公式:
    - Total: H[E_{W,ε}[p(y|x,W,ε)]]
    - Aleatoric: E_W[H[E_ε[p(y|x,W,ε)]]] ≈ 学习到的方差的函数
    - Epistemic: Total - Aleatoric

    Args:
        model: BSTGCN模型 (heteroscedastic=True)
        x: 输入数据 (batch, seq_len, features)
        num_samples: 权重采样次数 (用于epistemic)
        num_aleatoric_samples: 每个权重采样的噪声采样次数 (用于aleatoric)
        use_stable_entropy: 是否使用数值稳定的熵计算
        return_decomposition: 是否返回完整分解

    Returns:
        pred_mean: 平均预测概率 (batch, num_classes)
        pred_class: 预测类别 (batch,)
        epistemic_uncertainty: 认知不确定性 (batch,)
        aleatoric_uncertainty: 数据不确定性 (batch,)
        total_uncertainty: 总不确定性 (batch,)
        mean_variance: 平均学习到的方差 (batch, num_classes) - 仅当return_decomposition=True
    """
    model.eval()

    all_probs = []  # 存储所有(W, ε)组合的预测概率
    all_variances = []  # 存储学习到的方差
    weight_sample_probs = []  # 每个权重采样的平均概率

    with torch.no_grad():
        for _ in range(num_samples):
            # 采样权重 W ~ q(W)
            logits, log_variance = model(x, sample=True, return_variance=True)

            # 计算方差
            variance = F.softplus(log_variance)  # (B, C)
            std = torch.sqrt(torch.clamp(variance, min=1e-6, max=10.0))
            all_variances.append(variance)

            # 对每个权重采样，进行多次噪声采样
            probs_for_this_weight = []
            for _ in range(num_aleatoric_samples):
                # 采样噪声 ε ~ N(0, I)
                epsilon = torch.randn_like(logits)
                # 添加噪声: f' = mu + sigma * ε
                noisy_logits = logits + std * epsilon
                probs = F.softmax(noisy_logits, dim=-1)
                probs_for_this_weight.append(probs)
                all_probs.append(probs)

            # 该权重采样的平均概率
            weight_mean_probs = torch.stack(probs_for_this_weight, dim=0).mean(dim=0)
            weight_sample_probs.append(weight_mean_probs)

    # 聚合所有结果
    all_probs_stack = torch.stack(all_probs, dim=0)  # (M*K, B, C)
    weight_probs_stack = torch.stack(weight_sample_probs, dim=0)  # (M, B, C)
    mean_variance = torch.stack(all_variances, dim=0).mean(dim=0)  # (B, C)

    # 最终预测
    pred_mean = all_probs_stack.mean(dim=0)  # (B, C)
    pred_class = pred_mean.argmax(dim=-1)  # (B,)

    # 计算不确定性分解
    if use_stable_entropy:
        # Total Uncertainty: H[E_{W,ε}[p]]
        total_uncertainty = torch.special.entr(pred_mean).sum(dim=-1)  # (B,)

        # 每个(W,ε)采样的熵
        entropy_all_samples = torch.special.entr(all_probs_stack).sum(dim=-1)  # (M*K, B)

        # 每个权重采样的平均概率的熵
        entropy_weight_samples = torch.special.entr(weight_probs_stack).sum(dim=-1)  # (M, B)
    else:
        eps = 1e-10
        total_uncertainty = -torch.sum(pred_mean * torch.log(pred_mean + eps), dim=-1)
        entropy_all_samples = -torch.sum(all_probs_stack * torch.log(all_probs_stack + eps), dim=-1)
        entropy_weight_samples = -torch.sum(weight_probs_stack * torch.log(weight_probs_stack + eps), dim=-1)

    # 计算 E_W[H[E_ε[p]]] - 每个权重采样的平均概率的熵的平均
    expected_entropy_weight_mean = entropy_weight_samples.mean(dim=0)  # (B,)

    # 计算 E_W[E_ε[H[p]]] - 所有采样熵的平均
    expected_entropy_all = entropy_all_samples.mean(dim=0)  # (B,)

    # Epistemic: 权重不确定性导致的预测差异
    # = H[E_{W,ε}[p]] - E_W[H[E_ε[p]]]
    epistemic_uncertainty = total_uncertainty - expected_entropy_weight_mean  # (B,)

    # Aleatoric (熵分解): 数据噪声导致的预测差异
    # = E_W[H[E_ε[p]]] - E_W[E_ε[H[p]]]
    aleatoric_entropy = expected_entropy_weight_mean - expected_entropy_all  # (B,)

    # Aleatoric (直接方差): 使用学习到的方差的均值作为数据不确定性
    # 这是更直接的度量，因为方差直接反映了模型对数据噪声的估计
    aleatoric_variance = mean_variance.mean(dim=-1)  # (B,) - 所有类别方差的均值

    # 组合两种aleatoric度量: 使用方差作为主要度量
    # 因为熵分解的aleatoric可能很小，而方差是直接学习的
    aleatoric_uncertainty = aleatoric_variance

    # 确保非负 (数值误差可能导致微小负值)
    epistemic_uncertainty = torch.clamp(epistemic_uncertainty, min=0.0)
    aleatoric_uncertainty = torch.clamp(aleatoric_uncertainty, min=0.0)

    if return_decomposition:
        return (pred_mean, pred_class, epistemic_uncertainty,
                aleatoric_uncertainty, total_uncertainty, mean_variance)
    else:
        return pred_mean, pred_class, epistemic_uncertainty, aleatoric_uncertainty


def adaptive_mc_inference(
    model, x,
    min_samples=3,
    max_samples=8,
    confidence_threshold=0.95,
    uncertainty_threshold=0.05,
    use_stable_entropy=True
):
    """
    自适应蒙特卡洛推理 (优化版)

    根据初始采样的置信度动态决定是否继续采样

    Args:
        model: BSTGCN模型
        x: 输入数据
        min_samples: 最小采样次数 (默认3)
        max_samples: 最大采样次数 (默认8)
        confidence_threshold: 置信度阈值 (默认0.95)
        uncertainty_threshold: 不确定性阈值 (默认0.05)
        use_stable_entropy: 是否使用稳定熵计算

    Returns:
        pred_mean: 平均预测概率
        pred_class: 预测类别
        epistemic_uncertainty: 认知不确定性
        num_samples_used: 实际使用的采样次数
    """
    model.eval()
    probs_list = []

    with torch.no_grad():
        # 初始少量采样
        for _ in range(min_samples):
            logits = model(x, sample=True)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

        # 计算初始置信度
        probs_stack = torch.stack(probs_list, dim=0)
        pred_mean = probs_stack.mean(dim=0)
        max_prob = pred_mean.max(dim=-1)[0]

        # 如果置信度足够高，直接返回
        if max_prob.min() > confidence_threshold:
            pred_class = pred_mean.argmax(dim=-1)
            uncertainty = torch.zeros(x.size(0), device=x.device)
            return pred_mean, pred_class, uncertainty, min_samples

        # 否则继续采样
        for _ in range(max_samples - min_samples):
            logits = model(x, sample=True)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

            # 每次采样后检查
            probs_stack = torch.stack(probs_list, dim=0)
            pred_mean = probs_stack.mean(dim=0)

            # 计算不确定性
            if use_stable_entropy:
                entropy_mean = torch.special.entr(pred_mean).sum(dim=-1)
                entropy_samples = torch.special.entr(probs_stack).sum(dim=-1)
            else:
                entropy_mean = -torch.sum(pred_mean * torch.log(pred_mean + 1e-10), dim=-1)
                entropy_samples = -torch.sum(probs_stack * torch.log(probs_stack + 1e-10), dim=-1)

            mean_entropy = entropy_samples.mean(dim=0)
            uncertainty = entropy_mean - mean_entropy

            # 如果不确定性足够低，停止采样
            if uncertainty.max() < uncertainty_threshold:
                break

        # 最终结果
        probs_stack = torch.stack(probs_list, dim=0)
        pred_mean = probs_stack.mean(dim=0)
        pred_class = pred_mean.argmax(dim=-1)

        if use_stable_entropy:
            entropy_mean = torch.special.entr(pred_mean).sum(dim=-1)
            entropy_samples = torch.special.entr(probs_stack).sum(dim=-1)
        else:
            entropy_mean = -torch.sum(pred_mean * torch.log(pred_mean + 1e-10), dim=-1)
            entropy_samples = -torch.sum(probs_stack * torch.log(probs_stack + 1e-10), dim=-1)

        mean_entropy = entropy_samples.mean(dim=0)
        epistemic_uncertainty = entropy_mean - mean_entropy

        return pred_mean, pred_class, epistemic_uncertainty, len(probs_list)


def batch_mc_inference(model, dataloader, num_samples=30, device='cuda'):
    """
    批量蒙特卡洛推理

    对整个数据集进行MC推理

    Args:
        model: BSTGCN模型
        dataloader: 数据加载器
        num_samples: MC采样次数
        device: 设备

    Returns:
        all_pred_mean: 所有预测概率
        all_pred_class: 所有预测类别
        all_uncertainty: 所有不确定性
        all_labels: 所有真实标签
    """
    model.eval()

    all_pred_mean = []
    all_pred_class = []
    all_uncertainty = []
    all_labels = []

    for sequences, labels in dataloader:
        sequences = sequences.to(device)

        # MC推理
        pred_mean, pred_class, uncertainty = mc_inference(
            model, sequences, num_samples
        )

        all_pred_mean.append(pred_mean.cpu())
        all_pred_class.append(pred_class.cpu())
        all_uncertainty.append(uncertainty.cpu())
        all_labels.append(labels)

    # 拼接结果
    all_pred_mean = torch.cat(all_pred_mean, dim=0)
    all_pred_class = torch.cat(all_pred_class, dim=0)
    all_uncertainty = torch.cat(all_uncertainty, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_pred_mean, all_pred_class, all_uncertainty, all_labels

