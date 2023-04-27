"""
This code comes originally from https://github.com/rpryzant/causal-bert-pytorch
At the time of writing the original code contained an error
(https://github.com/rpryzant/causal-bert-pytorch/issues/6)
that made using one of the methods (.ATE()) unadvisable.
This version of code fixes this error.

An extensible implementation of the Causal Bert model from 
"Adapting Text Embeddings for Causal Inference" 
    (https://arxiv.org/abs/1905.12741)
"""
from collections import defaultdict
import os
import pickle

import scipy
from sklearn.model_selection import KFold

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertModel, BertPreTrainedModel, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from transformers import DistilBertTokenizer
from transformers import DistilBertModel, DistilBertPreTrainedModel

from torch.nn import CrossEntropyLoss

import torch
import torch.nn as nn
from scipy.special import softmax
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
import math

CUDA = (torch.cuda.device_count() > 0)
MASK_IDX = 103


def platt_scale(outcome, probs):
    logits = logit(probs)
    logits = logits.reshape(-1, 1)
    log_reg = LogisticRegression(penalty='none', warm_start=True, solver='lbfgs')
    log_reg.fit(logits, outcome)
    return log_reg.predict_proba(logits)


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def make_bow_vector(ids, vocab_size, use_counts=False):
    """ Make a sparse BOW vector from a tensor of dense ids.
    Args:
        ids: torch.LongTensor [batch, features]. Dense tensor of ids.
        vocab_size: vocab size for this tensor.
        use_counts: if true, the outgoing BOW vector will contain
            feature counts. If false, will contain binary indicators.
    Returns:
        The sparse bag-of-words representation of ids.
    """
    vec = torch.zeros(ids.shape[0], vocab_size)
    ones = torch.ones_like(ids, dtype=torch.float)
    if CUDA:
        vec = vec.cuda()
        ones = ones.cuda()
        ids = ids.cuda()

    vec.scatter_add_(1, ids, ones)
    vec[:, 1] = 0.0  # zero out pad
    if not use_counts:
        vec = (vec != 0).float()
    return vec



class CausalBert(DistilBertPreTrainedModel):
    """The model itself."""
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size

        self.distilbert = DistilBertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.Q_cls = nn.ModuleDict()

        for T in range(2):
            # ModuleDict keys have to be strings..
            self.Q_cls['%d' % T] = nn.Sequential(
                nn.Linear(config.hidden_size + self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, self.num_labels))

        self.g_cls = nn.Linear(config.hidden_size + self.num_labels, 
            self.config.num_labels)

        self.init_weights()

    def forward(self, W_ids, W_len, W_mask, C, T, Y=None, use_mlm=True):
        if use_mlm:
            W_len = W_len.unsqueeze(1) - 2 # -2 because of the +1 below
            mask_class = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
            mask = (mask_class(W_len.shape).uniform_() * W_len.float()).long() + 1 # + 1 to avoid CLS
            target_words = torch.gather(W_ids, 1, mask)
            mlm_labels = torch.ones(W_ids.shape).long() * -100
            if CUDA:
                mlm_labels = mlm_labels.cuda()
            mlm_labels.scatter_(1, mask, target_words)
            W_ids.scatter_(1, mask, MASK_IDX)

        outputs = self.distilbert(W_ids, attention_mask=W_mask)
        seq_output = outputs[0]
        pooled_output = seq_output[:, 0]
        # seq_output, pooled_output = outputs[:2]
        # pooled_output = self.dropout(pooled_output)

        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = CrossEntropyLoss()(
                prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = 0.0

        C_bow = make_bow_vector(C.unsqueeze(1), self.num_labels)
        inputs = torch.cat((pooled_output, C_bow), 1)

        # g logits
        g = self.g_cls(inputs)
        if Y is not None:  # TODO train/test mode, this is a lil hacky
            g_loss = CrossEntropyLoss()(g.view(-1, self.num_labels), T.view(-1))
        else:
            g_loss = 0.0

        # conditional expected outcome logits: 
        # run each example through its corresponding T matrix
        # TODO this would be cleaner with sigmoid and BCELoss, but less general 
        #   (and I couldn't get it to work as well)
        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)

        if Y is not None:
            T0_indices = (T == 0).nonzero().squeeze()
            Y_T1_labels = Y.clone().scatter(0, T0_indices, -100)

            T1_indices = (T == 1).nonzero().squeeze()
            Y_T0_labels = Y.clone().scatter(0, T1_indices, -100)

            Q_loss_T1 = CrossEntropyLoss()(
                Q_logits_T1.view(-1, self.num_labels), Y_T1_labels)
            Q_loss_T0 = CrossEntropyLoss()(
                Q_logits_T0.view(-1, self.num_labels), Y_T0_labels)

            Q_loss = Q_loss_T0 + Q_loss_T1
        else:
            Q_loss = 0.0

        sm = nn.Softmax(dim=1)
        Q0 = sm(Q_logits_T0)[:, 1]
        Q1 = sm(Q_logits_T1)[:, 1]
        g = sm(g)[:, 1]

        return g, Q0, Q1, g_loss, Q_loss, mlm_loss



class CausalBertWrapper:
    """Model wrapper in charge of training and inference."""

    def __init__(self, g_weight=1.0, Q_weight=0.1, mlm_weight=1.0,
        batch_size=32):
        self.model = CausalBert.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False)
        if CUDA:
            self.model = self.model.cuda()

        self.loss_weights = {
            'g': g_weight,
            'Q': Q_weight,
            'mlm': mlm_weight
        }
        self.batch_size = batch_size


    def train(self, texts, confounds, treatments, outcomes,
            learning_rate=2e-5, epochs=3):
        dataloader = self.build_dataloader(
            texts, confounds, treatments, outcomes)

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(dataloader) * epochs
        warmup_steps = total_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(epochs):
            losses = []
            self.model.train()
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                    if CUDA: 
                        batch = (x.cuda() for x in batch)
                    W_ids, W_len, W_mask, C, T, Y = batch
                    # while True:
                    self.model.zero_grad()
                    g, Q0, Q1, g_loss, Q_loss, mlm_loss = self.model(W_ids, W_len, W_mask, C, T, Y)
                    loss = self.loss_weights['g'] * g_loss + \
                            self.loss_weights['Q'] * Q_loss + \
                            self.loss_weights['mlm'] * mlm_loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    losses.append(loss.detach().cpu().item())
                # print(np.mean(losses))
                    # if step > 5: continue
        return self.model


    def inference(self, texts, confounds, outcome=None):
        self.model.eval()
        dataloader = self.build_dataloader(texts, confounds, outcomes=outcome,
            sampler='sequential')
        Q0s = []
        Q1s = []
        Ys = []
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if CUDA: 
                batch = (x.cuda() for x in batch)
            W_ids, W_len, W_mask, C, T, Y = batch
            g, Q0, Q1, _, _, _ = self.model(W_ids, W_len, W_mask, C, T, use_mlm=False)
            Q0s += Q0.detach().cpu().numpy().tolist()
            Q1s += Q1.detach().cpu().numpy().tolist()
            Ys += Y.detach().cpu().numpy().tolist()
            # if i > 5: break
        probs = np.array(list(zip(Q0s, Q1s)))
        preds = np.argmax(probs, axis=1)

        return probs, preds, Ys

    def ATE(self, C, W, Y=None, platt_scaling=False):
        Q_probs, _, Ys = self.inference(W, C, outcome=Y)
        if platt_scaling and Y is not None:
            Q0 = platt_scale(Ys, Q_probs[:, 0])[:, 0]
            Q1 = platt_scale(Ys, Q_probs[:, 1])[:, 1]
        else:
            Q0 = Q_probs[:, 0]
            Q1 = Q_probs[:, 1]

        return np.mean(Q1 - Q0)

    def build_dataloader(self, texts, confounds, treatments=None, outcomes=None,
        tokenizer=None, sampler='random'):
        def collate_CandT(data):
            # sort by (C, T), so you can get boundaries later
            # (do this here on cpu for speed)
            data.sort(key=lambda x: (x[1], x[2]))
            return data
        # fill with dummy values
        if treatments is None:
            treatments = [-1 for _ in range(len(confounds))]
        if outcomes is None:
            outcomes = [-1 for _ in range(len(treatments))]

        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased', do_lower_case=True)

        out = defaultdict(list)
        for i, (W, C, T, Y) in enumerate(zip(texts, confounds, treatments, outcomes)):
            # out['W_raw'].append(W)
            encoded_sent = tokenizer.encode_plus(W, add_special_tokens=True,
                max_length=128,
                truncation=True,
                pad_to_max_length=True)

            out['W_ids'].append(encoded_sent['input_ids'])
            out['W_mask'].append(encoded_sent['attention_mask'])
            out['W_len'].append(sum(encoded_sent['attention_mask']))
            out['Y'].append(Y)
            out['T'].append(T)
            out['C'].append(C)
            # if i > 100: break

        data = (torch.tensor(out[x]) for x in ['W_ids', 'W_len', 'W_mask', 'C', 'T', 'Y'])
        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
            # collate_fn=collate_CandT)

        return dataloader


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('testdata.csv')
    cb = CausalBertWrapper(batch_size=2,
        g_weight=0.1, Q_weight=0.1, mlm_weight=1)
    print(df.T)
    cb.train(df['text'], df['C'], df['T'], df['Y'], epochs=1)
    print(cb.ATE(df['C'], df.text, platt_scaling=True))








