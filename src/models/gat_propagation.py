"""Graph Attention Network for Influence Propagation

Implements GAT for modeling how information and influence
propagate through organizational authority graphs over time.

Based on:
- Veličković et al. "Graph Attention Networks" (2018)
- Yang et al. "Enhance Information Propagation for GNNs" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple


class GATInfluencePropagation(nn.Module):
    """
    Graph Attention Network for modeling influence propagation
    in organizational hierarchies.
    
    Features:
    - Multi-head attention to capture different influence patterns
    - Temporal unrolling to model cascade dynamics
    - Residual connections for gradient flow
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_timesteps: int = 3,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            num_timesteps: Number of temporal steps (T)
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_channels = hidden_dim
            else:
                in_channels = hidden_dim * num_heads
                
            # Last layer uses single head for simplicity
            out_heads = 1 if i == num_layers - 1 else num_heads
            
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=out_heads,
                    dropout=dropout,
                    add_self_loops=True,
                    concat=(i != num_layers - 1)  # Don't concat on last layer
                )
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * (num_heads if i < num_layers - 1 else 1))
            for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Temporal aggregation (combines information across time steps)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GAT with temporal unrolling.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes (influence weights) [num_edges, edge_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            - Updated node representations [num_nodes, hidden_dim]
            - Attention weights (optional) [num_edges] or None
        """
        batch_size = x.size(0)
        
        # Input projection
        h = self.input_proj(x)  # [num_nodes, hidden_dim]
        
        # Store temporal states
        temporal_states = []
        attention_weights_all = [] if return_attention else None
        
        # Temporal unrolling
        for t in range(self.num_timesteps):
            h_t = h.clone()
            
            # Pass through GAT layers
            for i, gat_layer in enumerate(self.gat_layers):
                h_residual = h_t
                
                # GAT forward
                if return_attention:
                    h_t, (edge_index_out, attn_weights) = gat_layer(
                        h_t,
                        edge_index,
                        return_attention_weights=True
                    )
                    if t == self.num_timesteps - 1:  # Save last timestep attention
                        attention_weights_all.append(attn_weights)
                else:
                    h_t = gat_layer(h_t, edge_index)
                
                # Activation
                h_t = F.elu(h_t)
                
                # Layer norm
                h_t = self.layer_norms[i](h_t)
                
                # Dropout
                h_t = self.dropout(h_t)
                
                # Residual connection
                if self.use_residual and i > 0:
                    # Project residual to match dimensions if needed
                    if h_residual.size(-1) != h_t.size(-1):
                        h_residual = F.adaptive_avg_pool1d(
                            h_residual.unsqueeze(0),
                            h_t.size(-1)
                        ).squeeze(0)
                    h_t = h_t + h_residual
            
            temporal_states.append(h_t)
            
            # Update h for next timestep
            h = h_t
        
        # Aggregate temporal states
        if len(temporal_states) > 1:
            # Stack temporal states: [num_timesteps, num_nodes, hidden_dim]
            temporal_stack = torch.stack(temporal_states, dim=0)
            
            # Permute to [num_nodes, num_timesteps, hidden_dim]
            temporal_stack = temporal_stack.permute(1, 0, 2)
            
            # Temporal attention aggregation
            h_final, _ = self.temporal_attn(
                temporal_stack,
                temporal_stack,
                temporal_stack
            )
            
            # Take mean across time or use last state
            h_final = h_final.mean(dim=1)  # [num_nodes, hidden_dim]
        else:
            h_final = temporal_states[0]
        
        if return_attention and attention_weights_all:
            # Return average attention across heads/layers
            avg_attention = torch.cat(attention_weights_all, dim=0).mean(dim=0)
            return h_final, avg_attention
        else:
            return h_final, None
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Attention weights [num_edges]
        """
        _, attention = self.forward(x, edge_index, return_attention=True)
        return attention

