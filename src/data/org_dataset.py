"""Organizational Response Dataset for H-POSG

This module implements the dataset for loading organizational structures,
recommendations, and multi-step agent responses.
"""

import torch
from torch.utils.data import Dataset
import json
import networkx as nx
from typing import Dict, List, Tuple, Optional


class OrganizationalDataset(Dataset):
    """
    Dataset for organizational response prediction.
    
    Each sample contains:
    - Organization description (text)
    - Organizational chart (graph)
    - Recommendation text
    - Agent responses across time steps
    - Authority weights
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_agents: int = 500,
        max_seq_length: int = 512,
        transform=None
    ):
        """
        Args:
            data_path: Path to the dataset directory
            split: 'train', 'val', or 'test'
            max_agents: Maximum number of agents to consider
            max_seq_length: Maximum sequence length for text
            transform: Optional transform to be applied
        """
        self.data_path = data_path
        self.split = split
        self.max_agents = max_agents
        self.max_seq_length = max_seq_length
        self.transform = transform
        
        # Load organizational scenarios
        self.scenarios = self._load_scenarios()
        
    def _load_scenarios(self) -> List[Dict]:
        """Load organizational scenarios from JSON"""
        file_path = f"{self.data_path}/{self.split}_scenarios.json"
        with open(file_path, 'r') as f:
            scenarios = json.load(f)
        return scenarios
    
    def __len__(self) -> int:
        return len(self.scenarios)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single organizational scenario.
        
        Returns:
            Dict containing:
                - org_text: Organizational description
                - recommendation_text: Recommendation text
                - authority_graph: NetworkX graph
                - agent_features: Agent attributes (power, role, level)
                - responses: Agent responses at each time step
                - labels: Ground truth responses
        """
        scenario = self.scenarios[idx]
        
        # Extract components
        org_text = scenario['organization']['description']
        recommendation_text = scenario['recommendation']['text']
        
        # Build authority graph
        authority_graph = self._build_authority_graph(scenario['organization'])
        
        # Extract agent features
        agent_features = self._extract_agent_features(scenario['organization'])
        
        # Extract multi-step responses
        responses = scenario.get('responses', {})
        
        # Labels (final responses)
        labels = self._extract_labels(scenario['organization'])
        
        sample = {
            'org_text': org_text,
            'recommendation_text': recommendation_text,
            'authority_graph': authority_graph,
            'agent_features': agent_features,
            'responses': responses,
            'labels': labels,
            'scenario_id': scenario.get('id', idx)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _build_authority_graph(self, org_data: Dict) -> nx.DiGraph:
        """
        Build authority graph from organizational data.
        
        Args:
            org_data: Organization structure data
            
        Returns:
            NetworkX directed graph with influence weights
        """
        G = nx.DiGraph()
        
        # Add nodes (agents)
        for agent_id, agent in enumerate(org_data.get('agents', [])):
            G.add_node(
                agent_id,
                role=agent.get('role', ''),
                power=agent.get('power', 0.5),
                level=agent.get('hierarchy_level', 0),
                department=agent.get('department', '')
            )
        
        # Add edges (authority relationships)
        for edge in org_data.get('authority_edges', []):
            source = edge['source']
            target = edge['target']
            weight = edge.get('influence_weight', 1.0)
            G.add_edge(source, target, weight=weight)
        
        return G
    
    def _extract_agent_features(self, org_data: Dict) -> torch.Tensor:
        """
        Extract agent-level features.
        
        Returns:
            Tensor of shape [num_agents, feature_dim]
        """
        agents = org_data.get('agents', [])
        features = []
        
        for agent in agents:
            feat = [
                agent.get('power', 0.5),
                agent.get('hierarchy_level', 0) / 10.0,  # Normalize
                agent.get('risk_tolerance', 0.5),
                agent.get('influence_score', 0.5)
            ]
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_labels(self, org_data: Dict) -> torch.Tensor:
        """
        Extract response labels.
        
        Response categories:
        0: Strongly Oppose
        1: Oppose
        2: Neutral
        3: Support
        4: Strongly Support
        
        Returns:
            Tensor of shape [num_agents] with class labels
        """
        agents = org_data.get('agents', [])
        labels = []
        
        response_map = {
            'strongly_oppose': 0,
            'oppose': 1,
            'neutral': 2,
            'support': 3,
            'strongly_support': 4
        }
        
        for agent in agents:
            response = agent.get('final_response', 'neutral')
            label = response_map.get(response, 2)  # Default to neutral
            labels.append(label)
        
        return torch.tensor(labels, dtype=torch.long)


def collate_org_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching organizational data.
    
    Handles variable-size graphs and agent counts.
    """
    # TODO: Implement batching logic for graphs
    # This will use PyTorch Geometric batching
    raise NotImplementedError("Graph batching to be implemented with PyG")

