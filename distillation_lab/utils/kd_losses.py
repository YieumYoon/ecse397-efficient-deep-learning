"""Knowledge Distillation loss functions.

This module implements the loss functions used in knowledge distillation:
1. KL divergence between soft targets (teacher and student logits)
2. Combined loss (hard targets + soft targets)
3. Optional feature-based distillation (FitNets style)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0
) -> torch.Tensor:
    """Compute KL divergence loss between teacher and student soft targets.
    
    The temperature scaling softens the probability distributions, making
    the "dark knowledge" (relative probabilities of incorrect classes) more
    visible to the student.
    
    Formula:
        KL(p_t^τ || p_s^τ) where τ is temperature
        
    Note: We multiply by τ² to balance the gradient magnitude when using
    temperature scaling.
    
    Args:
        student_logits: Raw logits from student model [batch_size, num_classes]
        teacher_logits: Raw logits from teacher model [batch_size, num_classes]
        temperature: Temperature for softening distributions (default: 4.0)
    
    Returns:
        KL divergence loss scaled by temperature squared
    """
    # Apply temperature scaling
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # Compute KL divergence: KL(teacher || student)
    # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    )
    
    # Scale by temperature squared (as in Hinton's paper)
    return kl_div * (temperature ** 2)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 4.0
) -> torch.Tensor:
    """Combined knowledge distillation loss.
    
    Formula:
        L_KD = α·L_CE(y, p_s) + (1-α)·τ²·KL(p_t^τ || p_s^τ)
        
    Where:
        - L_CE is cross-entropy with hard targets (ground truth labels)
        - KL is KL divergence with soft targets (teacher logits)
        - α balances between hard and soft targets
        - τ is temperature for distillation
    
    Args:
        student_logits: Raw logits from student [batch_size, num_classes]
        teacher_logits: Raw logits from teacher [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        alpha: Weight for hard target loss (1-alpha for soft target loss)
        temperature: Temperature for soft target distillation
    
    Returns:
        Combined distillation loss
    """
    # Hard target loss: standard cross-entropy with ground truth
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft target loss: KL divergence with teacher
    soft_loss = kl_divergence_loss(student_logits, teacher_logits, temperature)
    
    # Weighted combination
    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
    
    return total_loss


def feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Feature-based distillation loss (FitNets style).
    
    Aligns intermediate feature representations between teacher and student.
    This can be useful when the student should not only mimic the final
    predictions but also the internal representations.
    
    Args:
        student_features: Student intermediate features [batch, ...]
        teacher_features: Teacher intermediate features [batch, ...]
        normalize: If True, normalize features before computing loss
    
    Returns:
        MSE loss between feature representations
    """
    # Ensure features have the same shape
    if student_features.shape != teacher_features.shape:
        # If dimensions don't match, add a projection layer
        # This should ideally be done in the model architecture
        raise ValueError(
            f"Feature shape mismatch: student {student_features.shape} "
            f"vs teacher {teacher_features.shape}. "
            "Add a projection layer to match dimensions."
        )
    
    if normalize:
        # L2 normalize features
        student_features = F.normalize(student_features, p=2, dim=1)
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
    
    # MSE loss between features
    loss = F.mse_loss(student_features, teacher_features)
    
    return loss


def cosine_feature_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor
) -> torch.Tensor:
    """Cosine similarity based feature distillation loss.
    
    Alternative to MSE that focuses on the direction of feature vectors
    rather than their magnitude.
    
    Args:
        student_features: Student features [batch, features]
        teacher_features: Teacher features [batch, features]
    
    Returns:
        Cosine distance loss (1 - cosine_similarity)
    """
    # Flatten features if needed
    student_flat = student_features.flatten(1)
    teacher_flat = teacher_features.flatten(1)
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)
    
    # Return cosine distance (minimize 1 - similarity)
    loss = (1.0 - cosine_sim).mean()
    
    return loss


class DistillationLossWrapper(nn.Module):
    """Wrapper class for distillation loss with configurable parameters.
    
    This can be useful when you want to encapsulate the loss configuration
    and reuse it across training loops.
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        """Initialize distillation loss wrapper.
        
        Args:
            alpha: Weight for hard target loss
            temperature: Temperature for soft target distillation
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss.
        
        Args:
            student_logits: Student predictions
            teacher_logits: Teacher predictions
            labels: Ground truth labels
        
        Returns:
            Combined distillation loss
        """
        return distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=self.alpha,
            temperature=self.temperature
        )

