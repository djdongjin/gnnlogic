import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.net.base_net import Net
from codes.net.net_registry import choose_encoder_decoder, choose_model
from codes.utils.config import get_config


class DualModel(Net):
    """
    A general model that support two set of models (each one contains one encoder and one decoder)
    """
    def __init__(self, model_config, shared_embeddings=None):
        super(DualModel, self).__init__(model_config)
        self.model_config = model_config
        self.model1_config = get_config(model_config.model1_config_name).model
        self.model2_config = get_config(model_config.model2_config_name).model

        self.model1_encoder, self.model1_decoder = choose_model(self.model1_config)
        self.model2_encoder, self.model2_decoder = choose_model(self.model2_config)

        if model_config.dual.load_pretrained in [1, 2]:
            self.load_pretrained_model(model_config.dual.pretrained_prefix,
                                       model_config.dual.pretrained_exp_id,
                                       model_config.dual.load_pretrained)

    def forward(self, batch_pos_neg):
        return None

    def load_pretrained_model(self, model_prefix, exp_id, to_which=1):
        checkpoint_name = '{}_{}_checkpoint.pt'.format(model_prefix, exp_id)
        checkpoint_path = os.path.join(self.model_config.model_save_path,
                                       checkpoint_name)
        print(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            logging.info("loading model from checkpoint {}".format(checkpoint_name))
            checkpoint = torch.load(checkpoint_path)
            if to_which == 1:
                self.model1_encoder.load_state_dict(checkpoint['model.encoder'])
                self.model1_decoder.load_state_dict(checkpoint['model.decoder'])
            else:
                self.model2_encoder.load_state_dict(checkpoint['model.encoder'])
                self.model2_decoder.load_state_dict(checkpoint['model.decoder'])
        else:
            raise FileNotFoundError("Checkpoint not found")


class DualModelDecoder(Net):
    def __init__(self, which=1):
        self.decoder_no = which
        self.decoder = None

    def forward(self, batch, step_batch):
        return self.decoder(batch, step_batch)
