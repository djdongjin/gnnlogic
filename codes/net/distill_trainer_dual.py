# Trainer class to get the loss and predictions
import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from addict import Dict
from codes.net.batch import Batch
import pdb

from codes.net.net_registry import choose_encoder_decoder


class DistillTrainer:
    def __init__(self, model_config, encoder_model, decoder_model,
                 max_entity_id=0):
        """
        Generic trainer for an encoder-decoder model
        Changes:
            - now bringing the embedding handling logic to trainer itself,
            - so we can have a clear demarcation of concerns
        :param encoder_model:
        :param decoder_model:
        :param max_entity_id: max entity id
        """
        self.model_config = model_config
        self.max_entity_id = max_entity_id
        self.encoder_student = encoder_model
        self.decoder_student = decoder_model
        self.encoder_student.max_entity_id = max_entity_id
        self.decoder_student.max_entity_id = max_entity_id

        self.encoder_teacher = choose_encoder_decoder(model_config, model_config.dual.encoder_teacher)
        self.decoder_teacher = choose_encoder_decoder(model_config, model_config.dual.decoder_teacher)
        if torch.cuda.is_available():
            self.encoder_teacher.cuda()
            self.decoder_teacher.cuda()
        self.encoder_teacher.max_entity_id = max_entity_id
        self.decoder_teacher.max_entity_id = max_entity_id

        if model_config.dual.load_pretrained:
            self.load_teacher(model_config.dual.teacher_model_prefix, model_config.dual.teacher_exp_id)

        # initialize embeddings
        if self.model_config.embedding.entity_embedding_policy in ['random', 'fixed']:
            # randomize the entity embeddings first time
            self.encoder_student.randomize_entity_embeddings(padding=(self.model_config.dual.student_type == 'text'))
            self.encoder_teacher.randomize_entity_embeddings(padding=(self.model_config.dual.teacher_type == 'text'))
        else:
            print("Learning entity embeddings")

        loss_criteria = model_config.loss_criteria
        if loss_criteria == 'CE':
            self.criteria = nn.CrossEntropyLoss()
        elif loss_criteria == 'NLL':
            self.criteria = nn.NLLLoss()
        else:
            raise NotImplementedError("Provided loss criteria not implemented")

        # kd loss
        self.T = model_config.dual.kd_temperature
        self.alpha = model_config.dual.alpha
        self.beta = model_config.dual.beta
        self.gamma = model_config.dual.gamma

        self.kd_loss_criteria = KDLoss(self.T) if self.beta > 0 else None
        self.nce_loss_criteria = NCELoss() if self.gamma > 0 else None

    def match_ent_emb(self):
        """
        modify entity embeddings in student such that student model has exactly the same
        entity embeddings with teacher model.
        :return:
        """
        pad_teacher = 1 if self.model_config.dual.teacher_type == 'text' else 0
        pad_student = 1 if self.model_config.dual.student_type == 'text' else 0

        n_ent_emb_teacher = min(self.encoder_teacher.max_entity_id+pad_teacher, self.encoder_teacher.embedding.weight.shape[0]-pad_teacher)
        n_ent_emb_student = min(self.encoder_student.max_entity_id+pad_student, self.encoder_student.embedding.weight.shape[0]-pad_student)
        n_ent = min(n_ent_emb_teacher, n_ent_emb_student)

        with torch.no_grad():
            ent_emb_teacher = self.encoder_teacher.embedding.weight[pad_teacher:pad_teacher+n_ent]
            self.encoder_student.embedding.weight[pad_student:pad_student+n_ent] = ent_emb_teacher

    def get_optimizers(self):
        """Method to return the list of optimizers for the trainer"""
        optimizers = []

        model_params_encoder = self.encoder_student.get_model_params()
        model_params_decoder = self.decoder_student.get_model_params()
        model_params = model_params_encoder + model_params_decoder

        if model_params:
            if self.model_config.optimiser.name == "adam":
                optimizers.append(optim.Adam(model_params,
                                             lr=self.model_config.optimiser.learning_rate,
                                             weight_decay=self.model_config.optimiser.l2_penalty
                                             ))
            elif self.model_config.optimiser.name == 'sgd':
                optimizers.append(optim.SGD(model_params, lr=self.model_config.optimiser.learning_rate,
                                            weight_decay=self.model_config.optimiser.l2_penalty))
        if optimizers:
            if self.model_config.optimiser.scheduler_type == "exp":
                schedulers = list(map(lambda optimizer: optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer, gamma=self.model_config.optimiser.scheduler_gamma), optimizers))
            elif self.model_config.optimiser.scheduler_type == "plateau":
                schedulers = list(map(lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, mode="max", patience=self.model_config.optimiser.scheduler_patience,
                    factor=self.model_config.optimiser.scheduler_gamma, verbose=True), optimizers))

            return optimizers, schedulers
        return None

    def batchLoss(self, batch: Batch, batch_neg: Batch=None, mode='train'):
        """
        Run the Loss
        -> For classification experiments, use loss_type = `classify` in model.config
        -> For seq2seq experiments, use  loss_type = `seq2seq` in model.config
        :param batch - Batch class
        :param mode:
        :return:
        """
        if self.model_config.embedding.entity_embedding_policy == 'random':
            # randomize the entity embeddings at each epoch
            self.encoder_teacher.randomize_entity_embeddings(padding=(self.model_config.dual.teacher_type == 'text'))
            self.match_ent_emb()

        if self.model_config.embedding.entity_embedding_policy == 'fixed':
            # fix the random entity embeddings which were used before
            self.encoder_teacher.randomize_entity_embeddings(fixed=True, padding=(self.model_config.dual.teacher_type == 'text'))
            self.match_ent_emb()

        # --------------- teacher network --------------
        with torch.no_grad():
            encoder_outputs_teacher, encoder_hidden_teacher = self.encoder_teacher(batch)
            batch.encoder_outputs = encoder_outputs_teacher
            batch.encoder_hidden = encoder_hidden_teacher
            batch.encoder_model = self.encoder_teacher

            query_rep_teacher = self.decoder_teacher.calculate_query(batch)

            step_batch = Dict()
            step_batch.query_rep = query_rep_teacher
            logits_teacher, attn_teacher, hidden_rep_teacher = self.decoder_teacher(batch, step_batch)
            if logits_teacher.dim() > 2:
                logits_teacher = logits_teacher.squeeze(1)
        # --------------- teacher network --------------
        # --------------- student network --------------
        encoder_outputs_student, encoder_hidden_student = self.encoder_student(batch)

        batch.encoder_outputs = encoder_outputs_student
        batch.encoder_hidden = encoder_hidden_student
        batch.encoder_model = self.encoder_student

        query_rep = self.decoder_student.calculate_query(batch)  # query representation or question representation
        # batch.outp should be B x 1
        step_batch = Dict()
        step_batch.query_rep = query_rep
        logits_student, attn_student, hidden_rep_student = self.decoder_student(batch, step_batch)
        if logits_student.dim() > 2:
            logits_student = logits_student.squeeze(1)
        # --------------- student network --------------

        # ----------------- loss -----------------------
        # adapted according to https://github.com/NervanaSystems/distiller
        loss = self.criteria(logits_student, batch.target.squeeze(1))

        if self.kd_loss_criteria:
            kd_loss = self.kd_loss_criteria(logits_student, logits_teacher)
            batch.kd_loss = kd_loss.item()
        else:
            kd_loss = 0.

        # TODO: add nce loss calculation
        if self.nce_loss_criteria:
            nce_loss = self.nce_loss_criteria()
            batch.nce_loss = nce_loss.item()
        else:
            nce_loss = 0.

        overall_loss = self.alpha * loss + self.beta * kd_loss + self.gamma * nce_loss

        topv, topi = logits_student.data.topk(1)
        next_words = topi.squeeze(1)
        decoder_outp = next_words
        # confidence of classes
        conf = torch.exp(F.log_softmax(logits_student, dim=1))

        return decoder_outp, overall_loss, conf

    def train(self):
        self.encoder_student.train()
        self.decoder_student.train()
        self.encoder_teacher.train()
        self.decoder_teacher.train()

    def eval(self):
        self.encoder_student.eval()
        self.decoder_student.eval()
        self.encoder_teacher.eval()
        self.decoder_teacher.eval()

    def expand_hidden(self, hidden_rep, max_abs=5):
        return hidden_rep.unsqueeze(2).expand(-1, -1, max_abs, -1).contiguous()\
            .view(hidden_rep.size(0), -1, hidden_rep.size(-1))

    def load_teacher(self, model_prefix, exp_id):
        checkpoint_name = '{}_{}_checkpoint.pt'.format(model_prefix, exp_id)
        checkpoint_path = os.path.join(self.model_config.model_save_path,
                                       checkpoint_name)
        print(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            logging.info("loading teacher encoder and decoder from checkpoint {}".format(checkpoint_name))
            self.encoder_teacher.load_state_dict(checkpoint['model.encoder'])
            self.decoder_teacher.load_state_dict(checkpoint['model.decoder'])
            print('load teacher encoder and decoder from', checkpoint_name)
        else:
            raise FileNotFoundError("Checkpoint not found")


class KDLoss(nn.Module):

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.temp = T

    def forward(self, logits_student, logits_teacher):
        soft_log_probs = F.log_softmax(logits_student / self.temp, dim=1)
        soft_targets = F.softmax(logits_teacher / self.temp, dim=1)
        distillation_loss = F.kl_div(soft_log_probs, soft_targets.detach(), size_average=False) * (self.temp ** 2) / \
                            soft_targets.shape[0]

        return distillation_loss


class NCELoss(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        return None


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.tensor(n_hidden, n_hidden), requires_grad=True)
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features