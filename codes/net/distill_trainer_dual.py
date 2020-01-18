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
from codes.cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation

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

        self.checking_teacher_acc = False
        if model_config.dual.load_pretrained_teacher:
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

        if self.gamma > 0 and model_config.dual.corrupt:
            feat_dim_t = feat_dim_s = model_config.embedding.dim
            if model_config.embedding.emb_type == 'one-hot':
                feat_dim_t = model_config.unique_nodes
            if model_config.dual.use_query_rep:
                feat_dim_t *= 3
                feat_dim_s *= 3
            if model_config.encoder.bidirectional:
                feat_dim_s *= 2

            if model_config.dual.batch_contrastive:
                self.contrastive_loss_criteria = BatchContrastiveLoss(feat_dim_s, feat_dim_t, model_config.embedding.dim,
                                                                      use_query_rep=model_config.dual.use_query_rep,
                                                                      residual=True)
            else:
                self.contrastive_loss_criteria = ContrastiveLoss(feat_dim_s, feat_dim_t, model_config.embedding.dim,
                                                                 use_query_rep=model_config.dual.use_query_rep)
            if torch.cuda.is_available():
                self.contrastive_loss_criteria.cuda()
        else:
            self.contrastive_loss_criteria = None

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

        model_params_teacher = self.encoder_teacher.get_model_params() + self.decoder_teacher.get_model_params()

        if model_params:
            if self.model_config.optimiser.name == "adam":
                optimizers.append(optim.Adam(model_params,
                                             lr=self.model_config.optimiser.learning_rate,
                                             weight_decay=self.model_config.optimiser.l2_penalty
                                             ))
            elif self.model_config.optimiser.name == 'sgd':
                optimizers.append(optim.SGD(model_params, lr=self.model_config.optimiser.learning_rate,
                                            weight_decay=self.model_config.optimiser.l2_penalty))

        if model_params_teacher and self.model_config.dual.fine_tune_teacher:
            print('Teacher is being fine tuned.')
            if self.model_config.optimiser.name == "adam":
                optimizers.append(optim.Adam(model_params_teacher,
                                             lr=self.model_config.optimiser.learning_rate,
                                             weight_decay=self.model_config.optimiser.l2_penalty
                                             ))
            elif self.model_config.optimiser.name == 'sgd':
                optimizers.append(optim.SGD(model_params_teacher, lr=self.model_config.optimiser.learning_rate,
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

    def encoder_decoder(self, encoder, decoder, batch, **kwargs):
        if kwargs:
            enc_outp, enc_hid = encoder(batch, kwargs)
        else:
            enc_outp, enc_hid = encoder(batch)
        batch.encoder_outputs = enc_outp
        batch.encoder_hidden = enc_hid
        batch.encoder_model = encoder

        query_rep = decoder.calculate_query(batch)

        step_batch = Dict()
        step_batch.query_rep = query_rep
        logits, attn, hid_rep = decoder(batch, step_batch)
        feat, query_rep = batch.decoder_feat, batch.query_rep

        if logits.dim() > 2:
            logits = logits.squeeze(1)

        return logits, attn, hid_rep, feat, query_rep

    def batchLoss(self, batch: Batch):
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
            logits_t, attn_t, hid_rep_t, feat_t, query_rep_t = self.encoder_decoder(self.encoder_teacher,
                                                                                    self.decoder_teacher, batch)
            # generate corrupted feat rep
            if self.model_config.dual.corrupt and not self.model_config.dual.batch_contrastive:
                logits_t_neg, attn_t_neg, hid_rep_t_neg, feat_t_neg, query_rep_t_neg = self.encoder_decoder(
                                                                    self.encoder_teacher,
                                                                    self.decoder_teacher,
                                                                    batch,
                                                                    corrupt=True,
                                                                    corrupt_method=self.model_config.dual.corrupt_method)
        # --------------- teacher network --------------
        # --------------- student network --------------
        logits_s, attn_s, hid_rep_s, feat_s, query_rep_s = self.encoder_decoder(self.encoder_student,
                                                                                self.decoder_student,
                                                                                batch)
        # --------------- student network --------------

        # ----------------- loss -----------------------
        loss = self.criteria(logits_s, batch.target.squeeze(1))
        batch.supervised_loss = loss.item()

        if self.kd_loss_criteria:
            kd_loss = self.kd_loss_criteria(logits_s, logits_t)
            batch.kd_loss = kd_loss.item()
        else:
            kd_loss = 0.

        if self.contrastive_loss_criteria:
            if self.model_config.dual.batch_contrastive:
                contrastive_loss, contrastive_acc = self.contrastive_loss_criteria(feat_s, query_rep_s,
                                                                                   feat_t, query_rep_t)
            else:
                contrastive_loss, contrastive_acc = self.contrastive_loss_criteria(feat_s, query_rep_s,
                                                                                   feat_t, query_rep_t,
                                                                                   feat_t_neg, query_rep_t_neg)
            batch.contrastive_loss = contrastive_loss.item()
            batch.contrastive_acc = contrastive_acc.item()
        else:
            contrastive_loss = 0.

        overall_loss = self.alpha * loss + self.beta * kd_loss + self.gamma * contrastive_loss
        # print(overall_loss.item(), loss.item(), kd_loss.item(), contrastive_loss.item())
        if self.checking_teacher_acc:
            logits = logits_t
        else:
            logits = logits_s
        topv, topi = logits.data.topk(1)
        next_words = topi.squeeze(1)
        decoder_outp = next_words
        # confidence of classes
        conf = torch.exp(F.log_softmax(logits, dim=1))

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
        # self.temp = T
        self.T = T

    # def forward(self, logits_student, logits_teacher):
    #     soft_log_probs = F.log_softmax(logits_student / self.temp, dim=1)
    #     soft_targets = F.softmax(logits_teacher / self.temp, dim=1)
    #     distillation_loss = F.kl_div(soft_log_probs, soft_targets.detach(), size_average=False) * (self.temp ** 2) / \
    #                         soft_targets.shape[0]
    #
    #     return distillation_loss

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class MLP(nn.Module):
    def __init__(self, inp_dim, outp_dim, residual=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, outp_dim),
            nn.ReLU(),
            nn.Linear(outp_dim, outp_dim),
        )
        if residual:
            self.residual = nn.Linear(inp_dim, outp_dim)
        else:
            self.residual = None

    def forward(self, x):
        outp = self.mlp(x)
        if self.residual:
            outp = outp.relu() + self.residual(x)
        return outp


class BatchContrastiveLoss(nn.Module):

    def __init__(self, dim_s, dim_t, dim, use_query_rep=False, residual=True):
        super(BatchContrastiveLoss, self).__init__()
        self.use_query_rep = use_query_rep
        self.linear_s = MLP(dim_s, dim, residual)
        self.linear_t = MLP(dim_t, dim, residual)

    def forward(self, feat_s, query_rep_s, feat_t, query_rep_t):
        if self.use_query_rep:
            feat_s = torch.cat([feat_s, query_rep_s], dim=1)
            feat_t = torch.cat([feat_t, query_rep_t], dim=1)

        feat_s = self.linear_s(feat_s)
        feat_t = self.linear_t(feat_t)

        batch_size = feat_s.size(0)
        pos_mask = torch.eye(batch_size, device=feat_s.device)
        neg_mask = 1 - pos_mask

        res = torch.mm(feat_s, feat_t.t())

        measure = 'JSD'
        E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
        E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
        E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

        acc = torch.zeros(size=(1,))     # TODO: how to check discriminator acc
        # print(E_neg.item(), neg_mask.sum().item(), E_pos.item(), pos_mask.sum().item())
        return E_neg - E_pos, acc


class ContrastiveLoss(nn.Module):

    def __init__(self, dim_s, dim_t, dim, use_query_rep=False):
        super(ContrastiveLoss, self).__init__()
        self.discriminator = Discriminator(dim_s, dim_t, dim)
        self.loss = nn.BCELoss()
        self.user_query_rep = use_query_rep

    def forward(self, feat_s, query_rep_s, feat_t_pos, query_rep_t_pos, feat_t_neg, query_rep_t_neg):
        if self.user_query_rep:
            feat_s = torch.cat([feat_s, query_rep_s], dim=1)
            feat_t_pos = torch.cat([feat_t_pos, query_rep_t_pos], dim=1)
            feat_t_neg = torch.cat([feat_t_neg, query_rep_t_neg], dim=1)
        pos = self.discriminator(feat_s, feat_t_pos)
        neg = self.discriminator(feat_s, feat_t_neg)

        acc = torch.sum((pos >= 0.5) == torch.ones_like(pos)) / pos.size(0) + \
              torch.sum((neg < 0.5) == torch.ones_like(pos)) / neg.size(0)

        return self.loss(pos, torch.ones_like(pos)) + self.loss(neg, torch.ones_like(neg)), acc


class Discriminator(nn.Module):
    def __init__(self, dim_s, dim_t, dim):
        super(Discriminator, self).__init__()
        self.linear_s = nn.Linear(dim_s, dim)
        self.linear_t = nn.Linear(dim_t, dim)
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1)
        # self.reset_parameters()

    def uniform(self, size, param):
        bound = 1.0 / math.sqrt(size)
        if param is not None:
            nn.init.uniform_(param, -bound, bound)

    def reset_parameters(self):
        for nm, param in self.named_parameters():
            if param.requires_grad:
                if 'weight' in param:
                    size = param.size(0)
                    self.uniform(size, param)
                else:
                    nn.init.constant_(param, 0)

    def forward(self, feat_student, feat_teacher):
        f_s, f_t = self.linear_s(feat_student).relu(), self.linear_t(feat_teacher).relu()
        feat = torch.cat([f_s, f_t], dim=-1)
        return self.linear2(self.linear1(feat).relu()).sigmoid()
