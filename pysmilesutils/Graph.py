import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from utils import *
import math
import torch_geometric


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()

    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()


class ExtraGraphFeatureEmebedding(nn.Module):

    def __init__(self, max_n_nodes, max_weight, atom_weights, extra_feature_type, graph_hidden_dim, n_layers, use_2d):
        super(ExtraGraphFeatureEmebedding, self).__init__()
        self.max_n_nodes = max_n_nodes
        self.extra_features_type = extra_feature_type
        self.use_extra_molecular_feature = True if atom_weights is not None else False
        self.num_atom_type = len(atom_weights) if atom_weights is not None else 0
        self.use_2d = use_2d

        self.extra_features = ExtraFeatures(extra_features_type=extra_feature_type, max_n_nodes=max_n_nodes) if self.use_2d else None

        self.extra_molecular_features = ExtraMolecularFeatures(max_weight=max_weight, atom_weights=atom_weights) if self.use_extra_molecular_feature else None

        scale = 0
        if self.use_extra_molecular_feature:
            scale += 1

        if self.use_2d:
            if self.extra_features_type in {'all'}:
                scale += 11

            elif self.extra_features_type in {'cycles'}:
                scale += 5

            elif self.extra_features_type in {'eigenvalues'}:
                scale += 7

        self.graph_encoder = nn.Linear(scale, graph_hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, node_features, adj=None, edge_type=None, node_mask=None):

        spatial_graph_embedding = self.extra_features(edge_type, node_mask.squeeze(-1).bool()) if (self.extra_features is not None) and (adj is not None) else None

        molecular_graph_embedding = self.extra_molecular_features(node_features[:, :, :self.num_atom_type]) if (self.extra_molecular_features is not None) else None

        if (spatial_graph_embedding is not None) and (molecular_graph_embedding is not None):
            graph_embedding = torch.cat([spatial_graph_embedding, molecular_graph_embedding], dim=-1)

        elif (spatial_graph_embedding is None) and (molecular_graph_embedding is not None):
            graph_embedding = molecular_graph_embedding

        elif (spatial_graph_embedding is not None) and (molecular_graph_embedding is None):
            graph_embedding = spatial_graph_embedding

        else:
            return None

        graph_embedding = self.graph_encoder(graph_embedding)
        return graph_embedding

max_n_nodes = 100
max_weight = 10.0
atom_weights = [1.0, 2.0, 3.0, 4.0, 5.0]  # 假设有5种原子类型
extra_feature_type = 'all'  # 可选: 'all', 'cycles', 'eigenvalues'
graph_hidden_dim = 64
n_layers = 2
use_2d = True

model = ExtraGraphFeatureEmebedding(
    max_n_nodes=max_n_nodes,
    max_weight=max_weight,
    atom_weights=atom_weights,
    extra_feature_type=extra_feature_type,
    graph_hidden_dim=graph_hidden_dim,
    n_layers=n_layers,
    use_2d=use_2d
)

# 创建模拟输入数据
batch_size = 4
num_nodes = 30
num_features = len(atom_weights)  # 与atom_weights长度匹配

node_features = torch.rand(batch_size, num_nodes, num_features)
adj = torch.rand(batch_size, num_nodes, num_nodes)  # 邻接矩阵
edge_type = torch.randint(0, 3, (batch_size, num_nodes, num_nodes))  # 边类型
node_mask = torch.ones(batch_size, num_nodes, 1)  # 节点掩码

# 前向传播
output = model(node_features, adj, edge_type, node_mask)
print(f"输出形状: {output.shape if output is not None else None}")