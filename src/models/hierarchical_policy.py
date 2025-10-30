"""Hierarchical Policy Network for Agent Response Prediction

Implements policy networks that predict agent responses conditioned on:
- Organizational context
- Recommendation
- Neighbor influences from GNN
- Higher-level agent decisions (hierarchical conditioning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class HierarchicalPolicyNetwork(nn.Module):
    """
    Hierarchical policy network for predicting agent responses.
    
    Key features:
    - Hierarchical conditioning: lower-level agents observe higher-level decisions
    - Multi-modal input: text embeddings + graph features + agent attributes
    - 5-class output: strongly oppose, oppose, neutral, support, strongly support
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        text_embed_dim: int = 768,
        agent_feature_dim: int = 4,
        num_classes: int = 5,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_hierarchical_attention: bool = False
    ):
        """
        Args:
            hidden_dim: Hidden dimension for MLP
            text_embed_dim: Dimension of text embeddings from dual encoder
            agent_feature_dim: Dimension of agent-specific features
            num_classes: Number of response classes (default: 5)
            num_layers: Number of MLP layers
            dropout: Dropout probability
            use_hierarchical_attention: Use attention over higher-level agents
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.text_embed_dim = text_embed_dim
        self.agent_feature_dim = agent_feature_dim
        self.num_classes = num_classes
        self.use_hierarchical_attention = use_hierarchical_attention
        
        # Input dimension calculation:
        # hidden_dim (from GNN) + 2*text_embed_dim (rec + org) + agent_feature_dim
        input_dim = hidden_dim + 2 * text_embed_dim + agent_feature_dim
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        layer_dims = [2048, 1024, 512][:num_layers]
        
        prev_dim = input_dim
        for layer_dim in layer_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, layer_dim))
            prev_dim = layer_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_classes)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in layer_dims
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Hierarchical attention (optional)
        if use_hierarchical_attention:
            self.hier_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            # Projection for higher-level agent states
            self.hier_proj = nn.Linear(num_classes, hidden_dim)
    
    def forward(
        self,
        gnn_hidden: torch.Tensor,
        rec_embedding: torch.Tensor,
        org_embedding: torch.Tensor,
        agent_features: torch.Tensor,
        higher_level_predictions: Optional[torch.Tensor] = None,
        higher_level_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict agent responses.
        
        Args:
            gnn_hidden: Hidden states from GNN [num_agents, hidden_dim]
            rec_embedding: Recommendation embedding [batch_size, text_embed_dim]
            org_embedding: Organization embedding [batch_size, text_embed_dim]
            agent_features: Agent-specific features [num_agents, agent_feature_dim]
            higher_level_predictions: Predictions from higher-level agents
                                     [num_agents, num_higher_agents, num_classes]
            higher_level_mask: Mask for valid higher-level agents
                              [num_agents, num_higher_agents]
            
        Returns:
            Response logits [num_agents, num_classes]
        """
        num_agents = gnn_hidden.size(0)
        
        # Expand text embeddings to match number of agents
        rec_emb_expanded = rec_embedding.unsqueeze(0).expand(num_agents, -1)
        org_emb_expanded = org_embedding.unsqueeze(0).expand(num_agents, -1)
        
        # Concatenate all inputs
        x = torch.cat([
            gnn_hidden,
            rec_emb_expanded,
            org_emb_expanded,
            agent_features
        ], dim=-1)
        
        # Add hierarchical conditioning
        if higher_level_predictions is not None:
            if self.use_hierarchical_attention:
                # Attention over higher-level agents
                higher_features = self.hier_proj(higher_level_predictions)
                
                # Query: current agent; Key/Value: higher-level agents
                query = gnn_hidden.unsqueeze(1)  # [num_agents, 1, hidden_dim]
                
                hier_context, _ = self.hier_attention(
                    query,
                    higher_features,
                    higher_features,
                    key_padding_mask=~higher_level_mask if higher_level_mask is not None else None
                )
                
                x = torch.cat([x, hier_context.squeeze(1)], dim=-1)
            else:
                # Simple averaging of higher-level predictions
                if higher_level_mask is not None:
                    masked_preds = higher_level_predictions * higher_level_mask.unsqueeze(-1)
                    num_higher = higher_level_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                    avg_higher_preds = masked_preds.sum(dim=1) / num_higher
                else:
                    avg_higher_preds = higher_level_predictions.mean(dim=1)
                
                x = torch.cat([x, avg_higher_preds], dim=-1)
        
        # Pass through MLP
        for i, (mlp_layer, layer_norm) in enumerate(zip(self.mlp_layers, self.layer_norms)):
            x = mlp_layer(x)
            x = layer_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        logits = self.output_layer(x)
        
        return logits
    
    def predict(
        self,
        gnn_hidden: torch.Tensor,
        rec_embedding: torch.Tensor,
        org_embedding: torch.Tensor,
        agent_features: torch.Tensor,
        hierarchy_levels: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Hierarchical prediction: predict level-by-level from top to bottom.
        
        Args:
            gnn_hidden: Hidden states from GNN
            rec_embedding: Recommendation embedding
            org_embedding: Organization embedding
            agent_features: Agent features
            hierarchy_levels: Level of each agent [num_agents]
            edge_index: Authority graph edges
            
        Returns:
            Predicted responses [num_agents, num_classes]
        """
        num_agents = gnn_hidden.size(0)
        device = gnn_hidden.device
        
        # Initialize predictions
        all_predictions = torch.zeros(num_agents, self.num_classes, device=device)
        
        # Get unique levels sorted from highest to lowest
        unique_levels = torch.unique(hierarchy_levels, sorted=True, descending=True)
        
        # Predict level by level
        for level in unique_levels:
            # Get agents at this level
            level_mask = (hierarchy_levels == level)
            level_indices = torch.where(level_mask)[0]
            
            if len(level_indices) == 0:
                continue
            
            # Get higher-level predictions for each agent
            # (find supervisors/influencers from authority graph)
            higher_preds_list = []
            higher_masks_list = []
            
            for agent_idx in level_indices:
                # Find higher-level agents that influence this agent
                # (source nodes in edges pointing to this agent)
                incoming_edges = edge_index[1] == agent_idx
                source_agents = edge_index[0][incoming_edges]
                
                # Filter to only higher-level agents
                higher_agents = source_agents[hierarchy_levels[source_agents] > level]
                
                if len(higher_agents) > 0:
                    higher_preds = all_predictions[higher_agents]
                    # Pad to fixed size for batching
                    max_higher = 10  # Assume max 10 higher-level influencers
                    if len(higher_agents) < max_higher:
                        padding = torch.zeros(
                            max_higher - len(higher_agents),
                            self.num_classes,
                            device=device
                        )
                        higher_preds = torch.cat([higher_preds, padding], dim=0)
                        mask = torch.cat([
                            torch.ones(len(higher_agents), device=device),
                            torch.zeros(max_higher - len(higher_agents), device=device)
                        ])
                    else:
                        higher_preds = higher_preds[:max_higher]
                        mask = torch.ones(max_higher, device=device)
                else:
                    higher_preds = torch.zeros(10, self.num_classes, device=device)
                    mask = torch.zeros(10, device=device)
                
                higher_preds_list.append(higher_preds)
                higher_masks_list.append(mask)
            
            # Batch predictions for this level
            if higher_preds_list:
                higher_preds_batch = torch.stack(higher_preds_list)
                higher_masks_batch = torch.stack(higher_masks_list)
            else:
                higher_preds_batch = None
                higher_masks_batch = None
            
            # Predict for this level
            level_logits = self.forward(
                gnn_hidden[level_indices],
                rec_embedding,
                org_embedding,
                agent_features[level_indices],
                higher_preds_batch,
                higher_masks_batch
            )
            
            # Softmax to get probabilities
            level_probs = F.softmax(level_logits, dim=-1)
            
            # Store predictions
            all_predictions[level_indices] = level_probs
        
        return all_predictions

