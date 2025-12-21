"""
BSTGCN - 贝叶斯时空图神经网络

用于网络流量分类的贝叶斯深度学习框架
"""

from .bayesian_layers import BayesianLinear, BayesianConv1d
from .bayesian_components import (
    BayesianGraphSAGELayer,
    BayesianDepthwiseSeparableTCN,
    BayesianEfficientSTBlock,
    BayesianGatedAttentionPooling
)
from .bstgcn_model import BSTGCNModel, KNNGraphBuilder
from .elbo_loss import ELBOLoss, LabelSmoothingCrossEntropy, KLWeightScheduler
from .mc_inference import mc_inference, adaptive_mc_inference, batch_mc_inference
from .utils import (
    CosineWarmupScheduler,
    get_class_weights,
    evaluate_metrics,
    plot_confusion_matrix,
    plot_uncertainty_distribution
)

__version__ = '1.0.0'
__author__ = 'BSTGCN Team'

__all__ = [
    # 贝叶斯基础层
    'BayesianLinear',
    'BayesianConv1d',
    
    # 贝叶斯时空组件
    'BayesianGraphSAGELayer',
    'BayesianDepthwiseSeparableTCN',
    'BayesianEfficientSTBlock',
    'BayesianGatedAttentionPooling',
    
    # 模型
    'BSTGCNModel',
    'KNNGraphBuilder',
    
    # 损失函数
    'ELBOLoss',
    'LabelSmoothingCrossEntropy',
    'KLWeightScheduler',
    
    # MC推理
    'mc_inference',
    'adaptive_mc_inference',
    'batch_mc_inference',
    
    # 工具函数
    'CosineWarmupScheduler',
    'get_class_weights',
    'evaluate_metrics',
    'plot_confusion_matrix',
    'plot_uncertainty_distribution',
]

