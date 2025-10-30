"""Full H-POSG Model Integration

Integrates all components:
1. Dual encoders (text → embeddings)
2. GAT (influence propagation)
3. Hierarchical policy network (agent response prediction)

This is the complete end-to-end architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from ..representation.dual_encoder import DualEncoder
from .gat_propagation import GATInfluencePropagation
from .hierarchical_policy import HierarchicalPolicyNetwork


class HPOSGModel(nn.Module):
    """
    Hierarchical Partially Observable Stochastic Game Model
    
    Complete architecture for organizational response prediction
    combining text encoding, graph propagation, and hierarchical
    agent policy learning.
    """
    
    def __init__(
        self,
        # Dual encoder parameters
        text_model_name: str = 'roberta-large',
        text_embed_dim: int = 768,
        freeze_text_encoder: bool = False,
        
        # GAT parameters
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 4,
        gnn_num_heads: int = 8,
        gnn_num_timesteps: int = 3,
        
        # Policy network parameters
        policy_hidden_dim: int = 256,
        agent_feature_dim: int = 4,
        num_response_classes: int = 5,
        
        # Training parameters
        dropout: float = 0.2
    ):
        """
        Args:
            text_model_name: Pre-trained transformer name
            text_embed_dim: Text embedding dimension
            freeze_text_encoder: Whether to freeze text encoders
            gnn_hidden_dim: GNN hidden dimension
            gnn_num_layers: Number of GAT layers
            gnn_num_heads: Number of attention heads
            gnn_num_timesteps: Temporal unrolling steps
            policy_hidden_dim: Policy network hidden dimension
            agent_feature_dim: Agent feature dimension
            num_response_classes: Number of response categories
            dropout: Dropout probability
        """
        super().__init__()
        
        # Store config
        self.config = {
            'text_embed_dim': text_embed_dim,
            'gnn_hidden_dim': gnn_hidden_dim,
            'num_response_classes': num_response_classes
        }
        
        # Component 1: Dual Encoders
        self.dual_encoder = DualEncoder(
            model_name=text_model_name,
            embedding_dim=text_embed_dim,
            freeze_base=freeze_text_encoder
        )
        
        # Component 2: Graph Neural Network
        # Input to GNN: agent features + text embeddings
        gnn_input_dim = agent_feature_dim + 2 * text_embed_dim
        
        self.gnn = GATInfluencePropagation(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            num_timesteps=gnn_num_timesteps,
            dropout=dropout
        )
        
        # Component 3: Hierarchical Policy Network
        self.policy_network = HierarchicalPolicyNetwork(
            hidden_dim=gnn_hidden_dim,
            text_embed_dim=text_embed_dim,
            agent_feature_dim=agent_feature_dim,
            num_classes=num_response_classes,
            dropout=dropout
        )
    
    def forward(
        self,
        # Text inputs
        rec_input_ids: torch.Tensor,
        rec_attention_mask: torch.Tensor,
        org_input_ids: torch.Tensor,
        org_attention_mask: torch.Tensor,
        
        # Graph inputs
        agent_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        
        # Hierarchical inputs
        hierarchy_levels: Optional[torch.Tensor] = None,
        
        # Control flags
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through H-POSG model.
        
        Args:
            rec_input_ids: Recommendation token IDs
            rec_attention_mask: Recommendation attention mask
            org_input_ids: Organization token IDs
            org_attention_mask: Organization attention mask
            agent_features: Agent attributes [num_agents, agent_feature_dim]
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge attributes (influence weights)
            hierarchy_levels: Agent hierarchy levels [num_agents]
            return_embeddings: Return text embeddings
            return_attention: Return GNN attention weights
            
        Returns:
            Dictionary containing:
                - logits: Response logits [num_agents, num_classes]
                - rec_embedding: (optional) Recommendation embedding
                - org_embedding: (optional) Organization embedding
                - gnn_attention: (optional) Attention weights
        """
        # Step 1: Encode texts
        rec_emb, org_emb = self.dual_encoder(
            rec_input_ids,
            rec_attention_mask,
            org_input_ids,
            org_attention_mask
        )
        
        # Step 2: Prepare GNN input
        # Expand text embeddings to each agent
        num_agents = agent_features.size(0)
        rec_emb_expanded = rec_emb.unsqueeze(0).expand(num_agents, -1)
        org_emb_expanded = org_emb.unsqueeze(0).expand(num_agents, -1)
        
        # Concatenate with agent features
        gnn_input = torch.cat([
            agent_features,
            rec_emb_expanded,
            org_emb_expanded
        ], dim=-1)
        
        # Step 3: Graph propagation
        gnn_hidden, gnn_attention = self.gnn(
            gnn_input,
            edge_index,
            edge_attr,
            return_attention=return_attention
        )
        
        # Step 4: Predict responses
        if hierarchy_levels is not None:
            # Hierarchical prediction (level-by-level)
            predictions = self.policy_network.predict(
                gnn_hidden,
                rec_emb,
                org_emb,
                agent_features,
                hierarchy_levels,
                edge_index
            )
        else:
            # Standard prediction (parallel for all agents)
            logits = self.policy_network(
                gnn_hidden,
                rec_emb,
                org_emb,
                agent_features
            )
            predictions = logits
        
        # Prepare output
        output = {'logits': predictions}
        
        if return_embeddings:
            output['rec_embedding'] = rec_emb
            output['org_embedding'] = org_emb
        
        if return_attention and gnn_attention is not None:
            output['gnn_attention'] = gnn_attention
        
        return output
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        agent_responses_history: Optional[torch.Tensor] = None,
        lambda_consistency: float = 0.1,
        lambda_equilibrium: float = 0.05
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss:
        L = L_pred + λ1*L_consistency + λ2*L_equilibrium
        
        Args:
            logits: Predicted logits [num_agents, num_classes]
            labels: Ground truth labels [num_agents]
            agent_responses_history: Historical responses for consistency
            lambda_consistency: Weight for consistency loss
            lambda_equilibrium: Weight for equilibrium loss
            
        Returns:
            Dictionary with loss components
        """
        # Prediction loss (cross-entropy)
        loss_pred = nn.functional.cross_entropy(logits, labels)
        
        # Consistency loss (temporal consistency)
        if agent_responses_history is not None:
            # TODO: Implement temporal consistency loss
            loss_consistency = torch.tensor(0.0, device=logits.device)
        else:
            loss_consistency = torch.tensor(0.0, device=logits.device)
        
        # Equilibrium loss (encourage Nash equilibrium)
        # TODO: Implement equilibrium regularization
        loss_equilibrium = torch.tensor(0.0, device=logits.device)
        
        # Total loss
        total_loss = (
            loss_pred +
            lambda_consistency * loss_consistency +
            lambda_equilibrium * loss_equilibrium
        )
        
        return {
            'loss': total_loss,
            'loss_pred': loss_pred,
            'loss_consistency': loss_consistency,
            'loss_equilibrium': loss_equilibrium
        }
    
    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

