import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Dropout, LayerNorm, BatchNorm1d, LeakyReLU
from torch_geometric.nn import MessagePassing, global_mean_pool, TransformerConv
from torch_geometric.nn import MessagePassing, global_mean_pool

# GIN Convolution with Edge features
class GINConvE(MessagePassing):
    def __init__(self, emb_dim, edge_input_dim=7, dropout_edge_mlp=0.2, dropout_out=0.2):
        super().__init__(aggr="add")

        self.mlp = Sequential(
            Linear(emb_dim, 2 * emb_dim),
            LayerNorm(2 * emb_dim),
            LeakyReLU(0.15),
            Dropout(dropout_out),
            Linear(2 * emb_dim, emb_dim),
            LayerNorm(emb_dim)
        )

        self.edge_encoder = Sequential(
            Linear(edge_input_dim, emb_dim),
            LayerNorm(emb_dim),
            LeakyReLU(0.15),
            Dropout(dropout_edge_mlp),
            Linear(emb_dim, emb_dim),
            LeakyReLU(0.15),
            Linear(emb_dim, emb_dim)
        )

        self.eps = torch.nn.Parameter(torch.Tensor([1e-6]))

        self.dropout = Dropout(dropout_out)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        out = self.mlp((1 + self.eps) * x + out)
        # Residual connection + dropout
        return x + self.dropout(out)

    def message(self, x_j, edge_attr):
        # Gated fusion tra x_j ed edge_attr
        gate = torch.sigmoid(edge_attr)
        return gate * x_j + (1 - gate) * edge_attr

    def update(self, aggr_out):
        return aggr_out

class TransformerConvE(torch.nn.Module):
    def __init__(self, in_channels, emb_dim, heads=1, concat=True, 
                 dropout_transformer=0.2, edge_input_dim=7, dropout_edge_mlp=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.heads = heads
        self.concat = concat

        self.edge_encoder = Sequential(
            Linear(edge_input_dim, emb_dim),
            LeakyReLU(0.15),
            Linear(emb_dim, emb_dim)
        )
        if dropout_edge_mlp > 0:
            self.edge_encoder.append(torch.nn.Dropout(dropout_edge_mlp))

        # TransformerConv parameters
        if concat:
            out_per_head = emb_dim // heads
            transformer_output_dim = heads * out_per_head
        else:
            out_per_head = emb_dim
            transformer_output_dim = emb_dim
        
        self.transformer_conv = TransformerConv(
            in_channels,
            out_per_head,
            heads=heads,
            concat=concat,
            dropout=dropout_transformer,
            edge_dim=emb_dim,  # Edge embedding dimension
            root_weight=True,
            beta=True
        )

        if concat and transformer_output_dim != emb_dim:
            self.final_proj = Linear(transformer_output_dim, emb_dim)
        else:
            self.final_proj = torch.nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.transformer_conv(x, edge_index, edge_attr=edge_embedding)
        out = self.final_proj(out)
        return out

class uWuModel(torch.nn.Module):
    def __init__(self, emb_dim, edge_input_dim, num_classes, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.edge_input_dim = edge_input_dim

        self.input_proj = Linear(1, emb_dim)

        # 2 GINConvE layers
        self.gin_block1 = torch.nn.ModuleList([
            GINConvE(emb_dim, self.edge_input_dim, dropout_edge_mlp=dropout, dropout_out=dropout)
            for _ in range(2)
        ])
        self.bn1 = BatchNorm1d(emb_dim)

        # 1 TransformerConvE layer
        self.transformer1 = TransformerConvE(emb_dim, emb_dim, heads=2, concat=True,
                                               dropout_transformer=dropout, edge_input_dim=edge_input_dim, dropout_edge_mlp=dropout)
        self.bn_transformer1 = BatchNorm1d(emb_dim)

        # 2 GINConvE layers
        self.gin_block2 = torch.nn.ModuleList([
            GINConvE(emb_dim, edge_input_dim, dropout_edge_mlp=dropout, dropout_out=dropout)
            for _ in range(2)
        ])
        self.bn2 = BatchNorm1d(emb_dim)

        self.pool = global_mean_pool
        #self.classifier = Linear(emb_dim, num_classes)
        self.classifier = Sequential(
            Linear(emb_dim, emb_dim // 2),
            LeakyReLU(0.15),
            Dropout(dropout),
            Linear(emb_dim // 2, num_classes)
        )
        self.dropout = Dropout(dropout)

    
    def forward(self, x, edge_index, edge_attr, batch):
        if x is None:
            #x = torch.ones((edge_index.max().item() + 1, 1), device=edge_index.device)     --------------------------------------------------------------------------------------------
            num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else (batch.max().item() + 1 if batch.numel() > 0 else 1)
            x = torch.zeros((num_nodes, 1), device=edge_index.device)
        
        x = self.input_proj(x)  # (N, emb_dim)

        # First 2 GINConvE layers
        for conv in self.gin_block1:
            x = conv(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout(x)

        # 1 TransformerConvE layer
        residual = x
        x = self.transformer1(x, edge_index, edge_attr)
        x = self.bn_transformer1(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout(x)
        x = x + residual  # Residual connection

        # Next 2 GINConvE layers
        for conv in self.gin_block2:
            x = conv(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout(x)

        x = self.pool(x, batch)
        return self.classifier(x)