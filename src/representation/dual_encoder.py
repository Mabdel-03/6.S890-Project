"""Dual Encoder Architecture for Text-to-Representation Learning

This module implements separate encoders for:
1. Recommendation text → embedding
2. Organizational context → embedding

The encoders are trained with contrastive learning to align
compatible recommendation-organization pairs.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Optional


class RecommendationEncoder(nn.Module):
    """
    Encoder for recommendation text.
    
    Uses pre-trained transformer (BERT/RoBERTa) to encode
    recommendation descriptions into fixed-dimensional embeddings.
    """
    
    def __init__(
        self,
        model_name: str = 'roberta-large',
        embedding_dim: int = 768,
        freeze_base: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Output embedding dimension
            freeze_base: Whether to freeze base transformer weights
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Projection head for contrastive learning
        transformer_dim = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(transformer_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_dim)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode recommendation text.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        # Get transformer output
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to embedding space
        embeddings = self.projection(cls_embedding)
        
        # L2 normalize for contrastive learning
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class OrganizationalContextEncoder(nn.Module):
    """
    Encoder for organizational context.
    
    Encodes organizational structure, culture, and historical
    response patterns into fixed-dimensional embeddings.
    """
    
    def __init__(
        self,
        model_name: str = 'roberta-large',
        embedding_dim: int = 768,
        freeze_base: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Output embedding dimension
            freeze_base: Whether to freeze base transformer weights
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Projection head
        transformer_dim = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(transformer_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_dim)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode organizational context.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_embedding)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class DualEncoder(nn.Module):
    """
    Dual encoder architecture combining recommendation and
    organizational context encoders.
    
    Trained with contrastive learning to align compatible pairs.
    """
    
    def __init__(
        self,
        model_name: str = 'roberta-large',
        embedding_dim: int = 768,
        freeze_base: bool = False,
        shared_transformer: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Output embedding dimension
            freeze_base: Whether to freeze base transformer weights
            shared_transformer: Whether to share transformer weights
        """
        super().__init__()
        
        self.recommendation_encoder = RecommendationEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            freeze_base=freeze_base
        )
        
        if shared_transformer:
            # Share transformer weights but have separate projection heads
            self.org_encoder = OrganizationalContextEncoder(
                model_name=model_name,
                embedding_dim=embedding_dim,
                freeze_base=freeze_base
            )
            self.org_encoder.transformer = self.recommendation_encoder.transformer
        else:
            self.org_encoder = OrganizationalContextEncoder(
                model_name=model_name,
                embedding_dim=embedding_dim,
                freeze_base=freeze_base
            )
    
    def encode_recommendation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode recommendation text"""
        return self.recommendation_encoder(input_ids, attention_mask)
    
    def encode_organization(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode organizational context"""
        return self.org_encoder(input_ids, attention_mask)
    
    def forward(
        self,
        rec_input_ids: torch.Tensor,
        rec_attention_mask: torch.Tensor,
        org_input_ids: torch.Tensor,
        org_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both recommendation and organization.
        
        Returns:
            Tuple of (recommendation_embeddings, organization_embeddings)
        """
        rec_emb = self.encode_recommendation(rec_input_ids, rec_attention_mask)
        org_emb = self.encode_organization(org_input_ids, org_attention_mask)
        
        return rec_emb, org_emb
    
    def compute_similarity(
        self,
        rec_emb: torch.Tensor,
        org_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between recommendation and org embeddings.
        
        Args:
            rec_emb: [batch_size, embedding_dim]
            org_emb: [batch_size, embedding_dim]
            
        Returns:
            Similarity scores [batch_size, batch_size]
        """
        # Embeddings are already L2-normalized
        similarity = torch.matmul(rec_emb, org_emb.T)
        return similarity

