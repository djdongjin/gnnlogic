import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.net.base_net import Net
from codes.net.net_registry import choose_encoder


class TextGraphInfoMax(Net):
    """

    """
    def __init__(self, model_config, shared_embeddings=None):
        super(TextGraphInfoMax, self).__init__(model_config)

        self.model_config = model_config
        self.graph_encoder = choose_encoder(model_config, model_config.infomax.graph_encoder)
        self.text_encoder = choose_encoder(model_config, model_config.infomax.text_encoder)
        if model_config.encoder.bidirectional:
            self.discriminator = Discriminator(model_config.encoder.hidden_dim * 3)
        else:
            self.discriminator = Discriminator(model_config.encoder.hidden_dim * 2)

        if model_config.infomax.pretrained_gnn:
            self.load_graph_encoder(model_config.infomax.gnn_model_prefix, model_config.infomax.gnn_exp_id)
            for param in self.graph_encoder.parameters():
                param.require_grad = False

    def forward(self, batch_pos_neg):
        pos_batch, neg_batch = batch_pos_neg
        outp_text, hidden_text = self.text_encoder(pos_batch)
        text_emb = torch.mean(outp_text, 1)

        outp_graph_pos, hidden_graph_pos = self.graph_encoder(pos_batch)
        graph_emb_pos = torch.mean(outp_graph_pos, dim=1)
        all_pair = self.discriminator(text_emb, graph_emb_pos)
        if neg_batch:
            outp_graph_neg, hidden_graph_neg = self.graph_encoder(neg_batch)
            graph_emb_neg = torch.mean(outp_graph_neg, dim=1)
            neg_pair = self.discriminator(text_emb, graph_emb_neg)
            all_pair = torch.cat([all_pair, neg_pair], dim=0)
        if all_pair.dim() >= 2:
            all_pair = all_pair.squeeze()
        # print(outp_text.device)
        # print(hidden_text.device)
        # print(all_pair.device)
        return outp_text, hidden_text, all_pair

    def load_graph_encoder(self, model_prefix, exp_id):
        checkpoint_name = '{}_{}_checkpoint.pt'.format(model_prefix, exp_id)
        checkpoint_path = os.path.join(self.model_config.model_save_path,
                                       checkpoint_name)
        print(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            logging.info("loading graph encoder from checkpoint {}".format(checkpoint_name))
            checkpoint = torch.load(checkpoint_path)
            self.graph_encoder.load_state_dict(checkpoint['model.encoder'])
        else:
            raise FileNotFoundError("Checkpoint not found")


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        # self.weight = nn.Parameter(torch.Tensor(n_hidden, 1))
        # self.reset_parameters()
        self.linear = nn.Linear(n_hidden, 1)

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.linear.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, text_emb, graph_emb):
        # features = torch.matmul(features, torch.matmul(self.weight, summary))
        feats = torch.cat([text_emb, graph_emb], dim=1)
        logits = self.linear(feats)
        return logits
