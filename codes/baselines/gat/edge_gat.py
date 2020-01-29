# GAT with Edge features
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, scatter_
from codes.baselines.gat.inits import *
from codes.net.base_net import Net
import torch.nn as nn


class EdgeGINConv(MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(EdgeGINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # in the CLUTRR dataset, there is no self_loops
        # edge_index, _ = remove_self_loops(edge_index)
        x_i = torch.cat([x, torch.zeros(x.size(0), edge_attr.size(1)).to(x.device)], dim=1)
        aggregated = self.propagate(edge_index, edge_attr=edge_attr, x=x)
        out = self.nn((1 + self.eps) * x_i + aggregated)
        return out

    def message(self, x_j, edge_index, edge_attr):
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class EdgeGatConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_dim,
                 heads=1,
                 concat=False,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(EdgeGatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        self.edge_update = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        # x = N x in_channels , N=no of nodes
        # edge_index : 2 x E, E is the no of edges
        # edge_attr: E x edge_dim
        # this line is adding a self loop to the set of edges. if we have 200 edges and 300 nodes, then the new
        # edge count is 200+300 = 500
        # but, the later half of the added nodes (300) should not have any edge attributes
        # basically it performs this `edge_index = torch.cat([edge_index, loop], dim=1)`
        # here, we should also append a set of [node x zeros] in the edge_attr
        # maybe they add self loops in order to propagate the messages coming from the node itself?
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0) # (500, 10)

        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes, edge_attr):
        # x_i/x_j = E x out_channel, one message for each incoming edge
        # x_i and x_j are lifted tensors of shape E x heads x out_channels
        # our edge attributes are E x edge_dim
        # naive approach would be to append the edge dim to the messages
        # first, repeat the edge attribute for each head
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        # Compute attention coefficients.
        # N.B - only modification is the attention is now computed with the edge attributes
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1) # N x (out_channels + edge_dim)
        aggr_out = torch.mm(aggr_out, self.edge_update) # N x out_channels

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out # N x out_channels

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GatEncoder(Net):
    """
    Encoder which uses EdgeGatConv
    """

    def __init__(self,model_config, shared_embeddings=None):
        super(GatEncoder, self).__init__(model_config)

        # flag to enable one-hot embedding if needed
        self.graph_mode = True
        self.one_hot = self.model_config.embedding.emb_type == 'one-hot'
        if self.one_hot:
            self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.unique_nodes,
                                                embedding_dim=self.model_config.unique_nodes)
            self.embedding.weight = Parameter(torch.eye(self.model_config.unique_nodes))
            self.model_config.embedding.dim = self.model_config.unique_nodes
            self.model_config.graph.node_dim = self.model_config.unique_nodes
        else:
            self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.unique_nodes,
                                                embedding_dim=self.model_config.embedding.dim,
                                                max_norm=1)
            torch.nn.init.xavier_uniform_(self.embedding.weight)

        # learnable embeddings
        if self.model_config.graph.edge_dim_type == 'one-hot':
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types, model_config.edge_types)
            self.edge_embedding.weight = Parameter(torch.eye(self.model_config.edge_types))
            self.model_config.graph.edge_dim = self.model_config.edge_types
        else:
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types, model_config.graph.edge_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight)

        if self.model_config.graph.conv_layer == 'gat':
            self.layer1 = EdgeGatConv(self.model_config.embedding.dim, self.model_config.embedding.dim,
                                    self.model_config.graph.edge_dim, heads=self.model_config.graph.num_reads,
                                    dropout=self.model_config.graph.dropout)
            self.layer2 = EdgeGatConv(self.model_config.embedding.dim, self.model_config.embedding.dim,
                                    self.model_config.graph.edge_dim)
        elif self.model_config.graph.conv_layer == 'gin':
            node_dim, edge_dim = self.model_config.embedding.dim, self.model_config.graph.edge_dim
            self.layer1 = EdgeGINConv(nn.Sequential(nn.Linear(node_dim + edge_dim, node_dim + edge_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(node_dim + edge_dim, node_dim)))
            self.layer2 = EdgeGINConv(nn.Sequential(nn.Linear(node_dim + edge_dim, node_dim + edge_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(node_dim + edge_dim, node_dim)))
        else:
            raise NotImplementedError('check model_config.graph.conv! it should be in [gin, gat]')

    def forward(self, batch, corrupt=False, corrupt_method='edge'):
        data = batch.geo_batch
        x = self.embedding(data.x).squeeze(1) # N x node_dim
        edge_attr = self.edge_embedding(data.edge_attr).squeeze(1) # E x edge_dim
        if corrupt:
            if corrupt_method == 'edge' or corrupt_method == 'edge-in-batch':
                perm = torch.randperm(edge_attr.shape[0])
                edge_attr = edge_attr[perm]
            elif corrupt_method == 'edge-global':
                total_edge_type = self.edge_embedding.weight.shape[0]
                sampled_edge_type = torch.randint_like(data.edge_attr, low=0, high=total_edge_type)
                edge_attr = self.edge_embedding(sampled_edge_type).squeeze(1)
            elif corrupt_method == 'node':
                perm = torch.randperm(x.shape[0])
                x = x[perm]
            else:
                raise NotImplementedError('{} is not implemented as a corrupt method'.format(corrupt_method))
        for nr in range(self.model_config.graph.num_message_rounds):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.layer1(x, data.edge_index, edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.layer2(x, data.edge_index, edge_attr)
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.geo_slices, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        x = torch.cat(chunks, dim=0)
        return x, None


class GatDecoder(Net):
    """
    Compute the graph state with the query
    """
    def __init__(self, model_config):
        super(GatDecoder, self).__init__(model_config)
        input_dim = model_config.embedding.dim * 3
        if model_config.embedding.emb_type == 'one-hot':
            input_dim = self.model_config.unique_nodes * 3
        self.decoder2vocab = self.get_mlp(
            input_dim,
            model_config.target_size
        )

    def calculate_query(self, batch):
        """
        Extract the node embeddings using batch.query_edge
        :param batch:
        :return:
        """
        nodes = batch.encoder_outputs # B x node x dim
        query = batch.query_edge.squeeze(1).unsqueeze(2).repeat(1,1,nodes.size(2)) # B x num_q x dim
        query_emb = torch.gather(nodes, 1, query)
        return query_emb.view(nodes.size(0), -1) # B x (num_q x dim)

    def forward(self, batch, step_batch):
        query = step_batch.query_rep
        # pool the nodes
        # mean pooling
        node_avg = torch.mean(batch.encoder_outputs,1) # B x dim
        # concat the query
        node_cat = torch.cat((node_avg, query), -1) # B x (dim + dim x num_q)

        batch.decoder_feat = node_avg
        batch.query_rep = query

        return self.decoder2vocab(node_cat), None, None # B x num_vocab
