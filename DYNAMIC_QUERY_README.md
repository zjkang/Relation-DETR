# Dynamic Query Generation for DETR-based Object Detection

This implementation provides the dynamic query generation framework described in the paper "Dynamic Query Learning via Latent Patterns for Object Detection". The framework introduces three main components to improve object detection performance:

## Overview

The dynamic query generation framework consists of three key components:

1. **Content-Aware Weight Generator Module**: Dynamically generates weights conditioned on input image content
2. **Pattern-based Representation Module**: Learns compact base patterns and combines them with dynamic weights
3. **Quality-Aware One-to-Many Assignment Module**: Dynamically determines appropriate positive samples based on prediction quality

## Key Features

### 1. Content-Aware Weight Generator
- Multi-scale feature fusion with top-down approach
- Channel and Spatial Attention mechanisms
- Feature extraction with dilated convolutions
- Dynamic weight generation via MLP networks

### 2. Pattern-based Representation
- Learnable base patterns as fundamental building blocks
- Dynamic query construction via weighted pattern combination
- Convex combination constraints for stable training

### 3. Quality-Aware Assignment
- IoU-based quality scoring
- Dynamic positive sample selection
- Adaptive supervision signal strength

## Implementation Details

### File Structure
```
models/bricks/
├── dynamic_query_modules.py    # Core dynamic query generation modules
├── dino_transformer.py         # Modified DINO transformer with dynamic queries
└── ...

models/detectors/
└── dino.py                     # Modified DINO detector with new loss computation

configs/dino_dynamic/
└── dino_dynamic_resnet50_800_1333.py  # Configuration for dynamic DINO
```

### Core Modules

#### DynamicQueryGenerator
The main class that integrates all three components:
```python
dynamic_query_generator = DynamicQueryGenerator(
    embed_dim=256,
    num_queries=900,
    num_patterns=256,
    num_scales=4,
    gamma=0.5
)
```

#### Content-Aware Weight Generator
- `FeatureExtractor`: 1x1 conv + dilated conv + ReLU
- `MultiScaleFeatureFusion`: Top-down fusion with attention
- `WeightGenerator`: MLP-based dynamic weight generation

#### Pattern-based Representation
- `PatternBasedRepresentation`: Base patterns + FFN + weighted combination
- Learnable base patterns with diversity regularization

#### Quality-Aware Assignment
- `QualityAwareAssignment`: IoU-based quality scoring
- Dynamic positive sample selection based on prediction quality

## Usage

### 1. Basic Usage
```python
from models.bricks.dynamic_query_modules import DynamicQueryGenerator

# Initialize the dynamic query generator
dynamic_query_generator = DynamicQueryGenerator(
    embed_dim=256,
    num_queries=900,
    num_patterns=256,
    num_scales=4,
    gamma=0.5
)

# Generate dynamic queries from multi-scale features
content_queries, position_queries, dynamic_weights = dynamic_query_generator(multi_scale_features)

# Compute pattern diversity loss
diversity_loss = dynamic_query_generator.compute_diversity_loss()
```

### 2. Integration with DINO Transformer
```python
from models.bricks.dino_transformer import DINOTransformer

# Initialize transformer with dynamic query generation
transformer = DINOTransformer(
    encoder=encoder,
    decoder=decoder,
    num_classes=80,
    use_dynamic_queries=True,
    num_patterns=256,
    gamma=0.5
)
```

### 3. Training Configuration
```python
# Configuration parameters
model = dict(
    type='DINO',
    use_dynamic_queries=True,
    num_patterns=256,      # Number of base patterns
    gamma=0.5,             # Quality-aware assignment balance
    beta=0.2,              # Pattern diversity loss weight
    # ... other parameters
)
```

## Loss Functions

The framework introduces several new loss components:

### 1. Pattern Diversity Loss
Encourages diversity among base patterns:
```python
def pattern_diversity_loss(base_patterns):
    normalized_patterns = F.normalize(base_patterns, p=2, dim=-1)
    similarity_matrix = torch.matmul(normalized_patterns, normalized_patterns.t())
    # Remove diagonal and compute average absolute cosine similarity
    diversity_loss = torch.abs(similarity_matrix).sum() / num_pairs
    return diversity_loss
```

### 2. Quality-Aware One-to-Many Loss
Uses IoU-aware assignment for better supervision:
```python
quality_scores = ious - gamma * confidence
k_j = max(ceil(sum(top_k_scores)), min_positive_samples)
```

### 3. Complete Loss Function
```python
L_total = L_1:m + L_aux + β * L_div
```

Where:
- `L_1:m`: Quality-aware one-to-many assignment loss
- `L_aux`: Auxiliary Hungarian matching loss
- `L_div`: Pattern diversity loss with weight β

## Key Advantages

1. **Content Adaptation**: Queries adapt to input image characteristics
2. **Pattern Reusability**: Base patterns provide shared representations
3. **Quality-Aware Training**: Dynamic supervision based on prediction quality
4. **Improved Convergence**: Better training signals lead to faster convergence
5. **Flexible Architecture**: Can be integrated with various DETR-based models

## Experimental Results

Based on the paper, the dynamic query generation framework achieves:
- Improved convergence speed
- Better detection performance on challenging scenarios
- More stable training with quality-aware assignment
- Enhanced pattern diversity and representation learning

## Configuration Parameters

### Core Parameters
- `num_patterns`: Number of base patterns (default: 256)
- `gamma`: Balance factor for quality-aware assignment (default: 0.5)
- `beta`: Weight for pattern diversity loss (default: 0.2)

### Architecture Parameters
- `embed_dim`: Embedding dimension (default: 256)
- `num_queries`: Number of object queries (default: 900)
- `num_scales`: Number of feature scales (default: 4)

## Training Tips

1. **Pattern Initialization**: Base patterns are initialized with small random values
2. **Loss Balancing**: Adjust β to balance pattern diversity vs. detection performance
3. **Quality Threshold**: Tune γ for optimal quality-aware assignment
4. **Learning Rate**: Use lower learning rates for pattern-related parameters
5. **Regularization**: Monitor pattern usage to avoid collapse

## Future Extensions

1. **Position Query Generation**: Separate position and content query generation
2. **Hierarchical Patterns**: Multi-level pattern hierarchies
3. **Cross-Dataset Patterns**: Transferable patterns across domains
4. **Adaptive Pattern Count**: Dynamic number of patterns based on complexity

## Citation

If you use this implementation, please cite the original paper:
```bibtex
@article{dynamic_query_detr,
  title={Dynamic Query Learning via Latent Patterns for Object Detection},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This implementation follows the same license as the original DETR codebase.
