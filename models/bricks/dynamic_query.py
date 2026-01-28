import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from torchvision.ops import boxes as box_ops


class ChannelAttention(nn.Module):
    """Channel Attention Module - Simplified version inspired by SENet"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.avg_pool(x).view(b, c)  # [B, C]
        max_out = self.max_pool(x).view(b, c)  # [B, C]
        
        # Apply MLP to both pooled features
        avg_out = self.mlp(avg_out)  # [B, C]
        max_out = self.mlp(max_out)  # [B, C]
        
        attention = self.sigmoid(avg_out + max_out)  # [B, C]
        attention = attention.view(b, c, 1, 1)  # [B, C, 1, 1]
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention Module as described in Coordinate Attention"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, H, W]
        # Global average pooling and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        attention = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.conv(attention)  # [B, 1, H, W]
        attention = self.sigmoid(attention)
        
        return x * attention


class FeatureExtractor(nn.Module):
    """Feature Extractor with 1x1 conv, dilated conv, and ReLU"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        # 1x1 convolution for channel projection
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # 3x3 dilated convolution for spatial feature extraction
        self.conv3x3 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=dilation, 
            dilation=dilation,
            bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion with top-down fusion and attention mechanisms"""
    def __init__(self, embed_dim: int, num_scales: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        self.feature_extractors = nn.ModuleList([
            FeatureExtractor(embed_dim, embed_dim, dilation=2**i) 
            for i in range(num_scales)
        ])
        
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention(embed_dim),
                SpatialAttention()
            ) for _ in range(num_scales - 1)
        ])
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of feature maps [s2, s3, s4, s5] with shapes
                [(B, C, H2, W2), (B, C, H3, W3), (B, C, H4, W4), (B, C, H5, W5)]
        Returns:
            Enhanced feature Z from s2 scale: (B, C, H2, W2)
        """
        extracted_features = []
        for i, feat in enumerate(multi_scale_features):
            extracted_feat = self.feature_extractors[i](feat)
            extracted_features.append(extracted_feat)
        
        # Top-down fusion starting from the highest level
        fused_feature = extracted_features[-1]
        
        # Fuse from top to bottom (s5 -> s4 -> s3 -> s2)
        for i in range(len(extracted_features) - 2, -1, -1):  # i = 2, 1, 0
            target_shape = extracted_features[i].shape[2:]
            if fused_feature.shape[2:] != target_shape:
                fused_feature = F.interpolate(fused_feature, size=target_shape, 
                                            mode='bilinear', align_corners=False)
            fused_feature = fused_feature + extracted_features[i]
            
            fused_feature = self.fusion_blocks[i](fused_feature)
        
        return fused_feature  # Enhanced feature Z from s2 scale


class WeightGenerator(nn.Module):
    """Weight Generator Network with MLP, LayerNorm, and ReLU"""
    def __init__(self, embed_dim: int, num_queries: int, num_patterns: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_queries * num_patterns)
        )
        
    def forward(self, enhanced_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced_feature: (B, C, H, W) - enhanced feature from fusion
        Returns:
            Dynamic weights W: (B, num_queries, num_patterns)
        """
        # Global average pooling
        pooled_feat = F.adaptive_avg_pool2d(enhanced_feature, 1)  # (B, C, 1, 1)
        pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)  # (B, C)
        
        # Generate weights
        weights = self.weight_net(pooled_feat)  # (B, num_queries * num_patterns)
        weights = weights.view(-1, self.num_queries, self.num_patterns)  # (B, num_queries, num_patterns)
        weights = F.softmax(weights, dim=-1)
        
        return weights


class ContentAwareWeightGenerator(nn.Module):
    """Content-Aware Weight Generator Module"""
    def __init__(self, embed_dim: int, num_queries: int, num_patterns: int, num_scales: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        
        self.feature_fusion = MultiScaleFeatureFusion(embed_dim, num_scales)
        self.weight_generator = WeightGenerator(embed_dim, num_queries, num_patterns)
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of multi-scale feature maps from encoder
        Returns:
            Dynamic weights W: (B, num_queries, num_patterns)
        """
        enhanced_feature = self.feature_fusion(multi_scale_features)
        dynamic_weights = self.weight_generator(enhanced_feature)
        
        return dynamic_weights


class PatternBasedRepresentation(nn.Module):
    """Pattern-based Representation Module"""
    def __init__(self, embed_dim: int, num_queries: int, num_patterns: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        
        # Base patterns Q^P
        self.base_patterns = nn.Parameter(torch.randn(num_patterns, embed_dim))
        nn.init.normal_(self.base_patterns, std=0.02)
        # FFN for processing patterns
        self.pattern_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, dynamic_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dynamic_weights: (B, num_queries, num_patterns) - dynamic weights from weight generator
        Returns:
            content_queries: (B, num_queries, embed_dim) - content object queries Q^C
            position_queries: (B, num_queries, embed_dim) - position queries (same as content for now)
        """
        batch_size = dynamic_weights.size(0)
        processed_patterns = self.pattern_ffn(self.base_patterns)  # (num_patterns, embed_dim)
        
        if processed_patterns.device != dynamic_weights.device:
            processed_patterns = processed_patterns.to(dynamic_weights.device)

        content_queries = torch.matmul(dynamic_weights, processed_patterns)  # (B, num_queries, embed_dim)
        position_queries = content_queries.clone()
        
        return content_queries, position_queries


class QualityAwareAssignment(nn.Module):
    """Quality-Aware One-to-Many Assignment Module"""
    def __init__(self, gamma: float = 0.5, min_positive_samples: int = 1):
        super().__init__()
        self.gamma = gamma
        self.min_positive_samples = min_positive_samples
        
    def compute_quality_scores(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        Compute quality scores for prediction-ground truth pairs
        Args:
            predictions: (B, N, 5) - [x, y, w, h, conf] for each prediction
            ground_truths: (B, M, 4) - [x, y, w, h] for each ground truth
        Returns:
            quality_scores: (B, N, M) - quality score for each prediction-gt pair
        """
        batch_size, num_pred, _ = predictions.shape
        _, num_gt, _ = ground_truths.shape
        
        # Extract boxes and confidences
        pred_boxes = predictions[:, :, :4]  # (B, N, 4)
        pred_conf = predictions[:, :, 4:5]  # (B, N, 1)
        
        # Compute IoU between all prediction and ground truth boxes
        ious = self.compute_iou(pred_boxes, ground_truths)  # (B, N, M)
        
        # Compute quality scores: IoU - gamma * confidence
        quality_scores = ious - self.gamma * pred_conf.unsqueeze(-1)  # (B, N, M)
        
        return quality_scores
    
    def compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between boxes1 and boxes2 using torchvision's optimized implementation
        Args:
            boxes1: (B, N, 4) - [x, y, w, h] (center format)
            boxes2: (B, M, 4) - [x, y, w, h] (center format)
        Returns:
            ious: (B, N, M)
        """
        batch_size, num_pred, _ = boxes1.shape
        _, num_gt, _ = boxes2.shape
        
        # Convert to xyxy format for torchvision's box_iou function
        boxes1_xyxy = box_ops.box_convert(boxes1, in_fmt='cxcywh', out_fmt='xyxy')  # (B, N, 4)
        boxes2_xyxy = box_ops.box_convert(boxes2, in_fmt='cxcywh', out_fmt='xyxy')  # (B, M, 4)
        
        # Compute IoU for each batch
        ious = torch.zeros(batch_size, num_pred, num_gt, device=boxes1.device, dtype=boxes1.dtype)
        for b in range(batch_size):
            ious[b] = box_ops.box_iou(boxes1_xyxy[b], boxes2_xyxy[b])  # (N, M)
        
        return ious
    
    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor, 
                top_k: int = 5) -> torch.Tensor:
        """
        Perform quality-aware one-to-many assignment
        Args:
            predictions: (B, N, 5) - predictions with confidence
            ground_truths: (B, M, 4) - ground truth boxes
            top_k: maximum number of positive samples per ground truth
        Returns:
            assignment_mask: (B, N, M) - binary mask indicating assignments
        """
        quality_scores = self.compute_quality_scores(predictions, ground_truths)  # (B, N, M)
        
        batch_size, num_pred, num_gt = quality_scores.shape
        
        # For each ground truth, select top-k predictions
        assignment_mask = torch.zeros_like(quality_scores)
        
        for b in range(batch_size):
            for gt_idx in range(num_gt):
                # Get quality scores for this ground truth
                gt_scores = quality_scores[b, :, gt_idx]
                # Select top-k predictions
                k_j = min(top_k, num_pred)
                _, top_indices = torch.topk(gt_scores, k_j)
                # Assign top k_j predictions as positive
                for pred_idx in top_indices:
                    assignment_mask[b, pred_idx, gt_idx] = 1.0
        
        return assignment_mask


class PatternDiversityLoss(nn.Module):
    """Pattern Diversity Loss to encourage diversity among patterns"""
    def __init__(self):
        super().__init__()
        
    def forward(self, base_patterns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_patterns: (num_patterns, embed_dim) - base patterns Q^P
        Returns:
            diversity_loss: scalar tensor
        """
        # Normalize patterns
        normalized_patterns = F.normalize(base_patterns, p=2, dim=-1)  # (num_patterns, embed_dim)
        
        # Compute cosine similarity between all pairs
        similarity_matrix = torch.matmul(normalized_patterns, normalized_patterns.t())  # (num_patterns, num_patterns)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # Compute average absolute cosine similarity
        num_pairs = similarity_matrix.size(0) * (similarity_matrix.size(0) - 1)
        if num_pairs == 0:
            return torch.tensor(0.0, device=base_patterns.device, dtype=base_patterns.dtype)
        
        diversity_loss = torch.abs(similarity_matrix).sum() / num_pairs
        
        return diversity_loss


class DynamicQueryGenerator(nn.Module):
    """Complete Dynamic Query Generation Framework"""
    def __init__(self, embed_dim: int, num_queries: int, num_patterns: int, 
                 num_scales: int = 4, gamma: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        
        # Three main components
        self.content_aware_weight_generator = ContentAwareWeightGenerator(
            embed_dim, num_queries, num_patterns, num_scales
        )
        self.pattern_based_representation = PatternBasedRepresentation(
            embed_dim, num_queries, num_patterns
        )
        self.quality_aware_assignment = QualityAwareAssignment(gamma)
        
        # Loss modules
        self.pattern_diversity_loss = PatternDiversityLoss()
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dynamic query generation framework
        Args:
            multi_scale_features: List of multi-scale feature maps from encoder
        Returns:
            content_queries: (B, num_queries, embed_dim) - content object queries
            position_queries: (B, num_queries, embed_dim) - position queries  
            dynamic_weights: (B, num_queries, num_patterns) - dynamic weights
        """
        # Ensure all features are on the same device
        device = multi_scale_features[0].device
        multi_scale_features = [feat.to(device) for feat in multi_scale_features]
        
        # 1. Content-aware weight generation
        dynamic_weights = self.content_aware_weight_generator(multi_scale_features)
        
        # 2. Pattern-based representation
        content_queries, position_queries = self.pattern_based_representation(dynamic_weights)
        
        return content_queries, position_queries, dynamic_weights
    
    def compute_diversity_loss(self) -> torch.Tensor:
        """Compute pattern diversity loss"""
        base_patterns = self.pattern_based_representation.base_patterns
        return self.pattern_diversity_loss(base_patterns)
    
    def compute_quality_aware_assignment(self, predictions: torch.Tensor, 
                                       ground_truths: torch.Tensor) -> torch.Tensor:
        """Compute quality-aware assignment"""
        return self.quality_aware_assignment(predictions, ground_truths)
