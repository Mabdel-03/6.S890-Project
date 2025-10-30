"""Contrastive Learning Losses for Dual Encoder Training

Implements various contrastive objectives:
- NT-Xent (InfoNCE)
- Triplet Loss  
- Supervised Contrastive Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss.
    
    Used in SimCLR and dual encoder training.
    Maximizes agreement between positive pairs and minimizes
    agreement with negative pairs.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Scaling temperature for similarities
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            
        Returns:
            Loss scalar
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device
        
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # Compute similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for dual encoder training.
    
    Encourages anchor-positive distance < anchor-negative distance.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim]
            negative: [batch_size, embedding_dim]
            
        Returns:
            Loss scalar
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    Extends InfoNCE to handle multiple positives per anchor
    when labels are available.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Scaling temperature
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size] class labels
            
        Returns:
            Loss scalar
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss

