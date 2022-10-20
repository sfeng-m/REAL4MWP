# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np
import random
from scipy.stats import truncnorm

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from .file_utils import cached_path
from .loss import LabelSmoothingLoss
from utils.tool import isnumber

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 new_pos_ids=False,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 ffn_type=0,
                 label_smoothing=None,
                 num_qkv=0,
                 seg_emb=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.new_pos_ids = new_pos_ids
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.ffn_type = ffn_type
            self.label_smoothing = label_smoothing
            self.num_qkv = num_qkv
            self.seg_emb = seg_emb
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        if hasattr(config, 'new_pos_ids') and config.new_pos_ids:
            self.num_pos_emb = 4
        else:
            self.num_pos_emb = 1
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size*self.num_pos_emb)

        # self.num_equ_embeddings = nn.Embedding(
        #     config.num_equ_size, config.hidden_size)
        self.num_equ_embeddings = nn.Embedding(3, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, num_equ_ids=None, task_idx=None, is_analogy_input=False):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if is_analogy_input:
            words_embeddings = input_ids
        else:
            words_embeddings = self.word_embeddings(input_ids)  # 取出每个词的embedding向量 add by sfeng
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.num_pos_emb > 1:
            num_batch = position_embeddings.size(0)
            num_pos = position_embeddings.size(1)
            position_embeddings = position_embeddings.view(
                num_batch, num_pos, self.num_pos_emb, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if num_equ_ids is not None:
            num_equ_embeddings = self.num_equ_embeddings(num_equ_ids)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings + num_equ_embeddings
        else:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):  # bert/transformer的self attention部分 add by sfeng
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if hasattr(config, 'num_qkv') and (config.num_qkv > 1):
            self.num_qkv = config.num_qkv
        else:
            self.num_qkv = 1

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size*self.num_qkv)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size*self.num_qkv)
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size*self.num_qkv)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.uni_debug_flag = True if os.getenv(
            'UNI_DEBUG_FLAG', '') else False
        if self.uni_debug_flag:
            self.register_buffer('debug_attention_probs',
                                 torch.zeros((512, 512)))
        if hasattr(config, 'seg_emb') and config.seg_emb:
            self.b_q_s = nn.Parameter(torch.zeros(
                1, self.num_attention_heads, 1, self.attention_head_size))
            self.seg_emb = nn.Embedding(
                config.type_vocab_size, self.all_head_size)
        else:
            self.b_q_s = None
            self.seg_emb = None

    def transpose_for_scores(self, x, mask_qkv=None):
        if self.num_qkv > 1:
            sz = x.size()[:-1] + (self.num_qkv,
                                  self.num_attention_heads, self.all_head_size)
            # (batch, pos, num_qkv, head, head_hid)
            x = x.view(*sz)
            if mask_qkv is None:
                x = x[:, :, 0, :, :]
            elif isinstance(mask_qkv, int):
                x = x[:, :, mask_qkv, :, :]
            else:
                # mask_qkv: (batch, pos)
                if mask_qkv.size(1) > sz[1]:
                    mask_qkv = mask_qkv[:, :sz[1]]
                # -> x: (batch, pos, head, head_hid)
                x = x.gather(2, mask_qkv.view(sz[0], sz[1], 1, 1, 1).expand(
                    sz[0], sz[1], 1, sz[3], sz[4])).squeeze(2)
        else:
            sz = x.size()[:-1] + (self.num_attention_heads,
                                  self.attention_head_size)
            # (batch, pos, head, head_hid)
            x = x.view(*sz)
        # (batch, head, pos, head_hid)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None, mask_qkv=None, seg_ids=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(hidden_states, self.key.weight)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(x_states, self.key.weight)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, mask_qkv)
        key_layer = self.transpose_for_scores(mixed_key_layer, mask_qkv)
        value_layer = self.transpose_for_scores(mixed_value_layer, mask_qkv)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if self.seg_emb is not None:
            seg_rep = self.seg_emb(seg_ids)
            # (batch, pos, head, head_hid)
            seg_rep = seg_rep.view(seg_rep.size(0), seg_rep.size(
                1), self.num_attention_heads, self.attention_head_size)
            qs = torch.einsum('bnih,bjnh->bnij',
                              query_layer+self.b_q_s, seg_rep)
            attention_scores = attention_scores + qs

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # 根据attention_mask决定要使用哪种训练方式(l2r/seq2seq) add by sfeng

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Q、K构造attention输出概率，多头注意力进行叠加
        attn_dist = nn.Softmax(dim=-1)(torch.sum(attention_scores, dim=1))

        if self.uni_debug_flag:
            _pos = attention_probs.size(-1)
            self.debug_attention_probs[:_pos, :_pos].copy_(
                attention_probs[0].mean(0).view(_pos, _pos))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attn_dist


class BertSelfOutput(nn.Module):  # bert/transformer的self attention 后的add&norm部分 add by sfeng
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None, mask_qkv=None, seg_ids=None):
        self_output, attn_dist = self.self(
            input_tensor, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attn_dist


class BertIntermediate(nn.Module):  # bert/transformer的feed forward部分 add by sfeng
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):  # bert/transformer的feed forward后的add&norm部分 add by sfeng
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerFFN(nn.Module):
    def __init__(self, config):
        super(TransformerFFN, self).__init__()
        self.ffn_type = config.ffn_type
        assert self.ffn_type in (1, 2)
        if self.ffn_type in (1, 2):
            self.wx0 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (2,):
            self.wx1 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (1, 2):
            self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        if self.ffn_type in (1, 2):
            x0 = self.wx0(x)
            if self.ffn_type == 1:
                x1 = x
            elif self.ffn_type == 2:
                x1 = self.wx1(x)
            out = self.output(x0 * x1)
        out = self.dropout(out)
        out = self.LayerNorm(out + x)
        return out


class BertLayer(nn.Module):  # 整个bert模块（一层）（即transformer的encode部分）add by sfeng
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.ffn_type = config.ffn_type
        if self.ffn_type:
            self.ffn = TransformerFFN(config)
        else:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None, mask_qkv=None, seg_ids=None):
        attention_output, attn_dist = self.attention(
            hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
        if self.ffn_type:
            layer_output = self.ffn(attention_output)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attn_dist


class BertEncoder(nn.Module):  # bert的encode部分，即多层的bert模块 add by sfeng
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, analogy_attention_mask, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, seg_ids=None):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        all_encoder_layers = []
        attn_dists = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                if i < int(self.num_hidden_layers/2):  # important, add by sfeng; (split for two part, first for question inner attention, second for question outer attention)
                    hidden_states, attn_dist = layer_module(
                        hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
                else:
                    hidden_states, attn_dist = layer_module(
                        hidden_states, analogy_attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                    attn_dists.append(attn_dist)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for i, layer_module in enumerate(self.layer):
                if i < int(self.num_hidden_layers/2):
                    hidden_states, attn_dist = layer_module(
                        hidden_states, attention_mask, mask_qkv=mask_qkv, seg_ids=seg_ids)
                else:
                    hidden_states, attn_dist = layer_module(
                        hidden_states, analogy_attention_mask, mask_qkv=mask_qkv, seg_ids=seg_ids)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                    attn_dists.append(attn_dist)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attn_dists


class BertPooler(nn.Module):  # 第一个token接全连接层，得到输出结果
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor
        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(
                self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        if pooled_output is None:
            seq_relationship_score = None
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):  # 根据输入参数的维度，对bert原本的参数做调整，使其适配新的输入（维度方面）； add by sfeng
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.copy_(torch.Tensor(
                truncnorm.rvs(-1, 1, size=list(module.weight.data.shape)) * self.config.initializer_range))
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name == 'bert-base-chinese' and os.path.isdir('../pytorch_pretrained_bert/bert_parameter/bert-base-chinese'):
            serialization_dir = '../pytorch_pretrained_bert/bert_parameter/bert-base-chinese'
            tempdir = None
        elif pretrained_model_name == 'bert-base-uncased' and os.path.isdir('../pytorch_pretrained_bert/bert_parameter/bert-base-uncased'):
            serialization_dir = '../pytorch_pretrained_bert/bert_parameter/bert-base-uncased'
            tempdir = None
        else:
            if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
                archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
            else:
                archive_file = pretrained_model_name
            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file, cache_dir=cache_dir)
            except FileNotFoundError:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name,
                        ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                        archive_file))
                return None
            if resolved_archive_file == archive_file:
                logger.info("loading archive file {}".format(archive_file))
            else:
                logger.info("loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
            tempdir = None
            if os.path.isdir(resolved_archive_file):
                serialization_dir = resolved_archive_file
            else:
                # Extract archive to temp dir
                tempdir = tempfile.mkdtemp()
                logger.info("extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir))
                with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(archive, tempdir)
                serialization_dir = tempdir
        # Load config
        if ('config_path' in kwargs) and kwargs['config_path']:
            config_file = kwargs['config_path']
        else:
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if 'type_vocab_size' in kwargs:
            config.type_vocab_size = kwargs['type_vocab_size']
        # define new relax_projection
        if ('relax_projection' in kwargs) and kwargs['relax_projection']:
            config.relax_projection = kwargs['relax_projection']
        # new position embedding
        if ('new_pos_ids' in kwargs) and kwargs['new_pos_ids']:
            config.new_pos_ids = kwargs['new_pos_ids']
        # define new relax_projection
        if ('task_idx' in kwargs) and kwargs['task_idx']:
            config.task_idx = kwargs['task_idx']
        # define new max position embedding for length expansion
        if ('max_position_embeddings' in kwargs) and kwargs['max_position_embeddings']:
            config.max_position_embeddings = kwargs['max_position_embeddings']
        # use fp32 for embeddings
        if ('fp32_embedding' in kwargs) and kwargs['fp32_embedding']:
            config.fp32_embedding = kwargs['fp32_embedding']
        # type of FFN in transformer blocks
        if ('ffn_type' in kwargs) and kwargs['ffn_type']:
            config.ffn_type = kwargs['ffn_type']
        # label smoothing
        if ('label_smoothing' in kwargs) and kwargs['label_smoothing']:
            config.label_smoothing = kwargs['label_smoothing']
        # dropout
        if ('hidden_dropout_prob' in kwargs) and kwargs['hidden_dropout_prob']:
            config.hidden_dropout_prob = kwargs['hidden_dropout_prob']
        if ('attention_probs_dropout_prob' in kwargs) and kwargs['attention_probs_dropout_prob']:
            config.attention_probs_dropout_prob = kwargs['attention_probs_dropout_prob']
        # different QKV
        if ('num_qkv' in kwargs) and kwargs['num_qkv']:
            config.num_qkv = kwargs['num_qkv']
        # segment embedding for self-attention
        if ('seg_emb' in kwargs) and kwargs['seg_emb']:
            config.seg_emb = kwargs['seg_emb']
        # initialize word embeddings
        _word_emb_map = None
        if ('word_emb_map' in kwargs) and kwargs['word_emb_map']:
            _word_emb_map = kwargs['word_emb_map']

        # add source vocab by sfeng;
        if ('source_vocab' in kwargs) and kwargs['source_vocab']:
            _source_vocab = kwargs['source_vocab']
        else:
            _source_vocab = None
        
        if ('primary_vocabs' in kwargs) and kwargs['primary_vocabs']:
            _primary_vocabs = kwargs['primary_vocabs']
            config.vocab_size = len(_primary_vocabs) + len(_source_vocab) if _source_vocab is not None else len(_primary_vocabs)

        # logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in ('config_path', 'type_vocab_size', 'relax_projection', 'new_pos_ids', 'task_idx', 'max_position_embeddings', 
                        'fp32_embedding', 'ffn_type', 'label_smoothing', 'hidden_dropout_prob', 'attention_probs_dropout_prob', 
                        'num_qkv', 'seg_emb', 'word_emb_map', 'source_vocab', 'primary_vocabs'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.  
        model = cls(config, *inputs, **kwargs)  # 初始化模型参数（随机初始化，此处尚未加载bert预训练参数） by sfeng;
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = 'bert.embeddings.token_type_embeddings.weight'
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
                config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                # state_dict[_k].data = state_dict[_k].data.resize_(config.type_vocab_size, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.type_vocab_size, state_dict[_k].shape[1])
                # # L2R
                # if config.type_vocab_size >= 3:
                #     state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                # # R2L
                # if config.type_vocab_size >= 4:
                #     state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                # add analogy embedding
                if config.type_vocab_size >= 4:
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[1, :])
                # S2S
                if config.type_vocab_size >= 6:
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
                # if config.type_vocab_size >= 7:
                #     state_dict[_k].data[6, :].copy_(state_dict[_k].data[1, :])
                # add analogy embedding
                if config.type_vocab_size >= 8:
                    state_dict[_k].data[6, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[7, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.type_vocab_size, :]

        _k = 'bert.embeddings.position_embeddings.weight'
        n_config_pos_emb = 4 if config.new_pos_ids else 1
        if (_k in state_dict) and (n_config_pos_emb*config.hidden_size != state_dict[_k].shape[1]):
            logger.info("n_config_pos_emb*config.hidden_size != state_dict[bert.embeddings.position_embeddings.weight] ({0}*{1} != {2})".format(
                n_config_pos_emb, config.hidden_size, state_dict[_k].shape[1]))
            assert state_dict[_k].shape[1] % config.hidden_size == 0
            n_state_pos_emb = int(state_dict[_k].shape[1]/config.hidden_size)
            assert (n_state_pos_emb == 1) != (n_config_pos_emb ==
                                              1), "!!!!n_state_pos_emb == 1 xor n_config_pos_emb == 1!!!!"
            if n_state_pos_emb == 1:
                state_dict[_k].data = state_dict[_k].data.unsqueeze(1).repeat(
                    1, n_config_pos_emb, 1).reshape((config.max_position_embeddings, n_config_pos_emb*config.hidden_size))
            elif n_config_pos_emb == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                state_dict[_k].data = state_dict[_k].data.view(
                    config.max_position_embeddings, n_state_pos_emb, config.hidden_size).select(1, _task_idx)

        # initialize new position embeddings
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info("config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
                config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                # state_dict[_k].data = state_dict[_k].data.resize_(config.max_position_embeddings, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(
                        old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start:start+chunk_size,
                                        :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        # initialize relax projection
        _k = 'cls.predictions.transform.dense.weight'
        n_config_relax = 1 if (config.relax_projection <
                               1) else config.relax_projection
        if (_k in state_dict) and (n_config_relax*config.hidden_size != state_dict[_k].shape[0]):
            logger.info("n_config_relax*config.hidden_size != state_dict[cls.predictions.transform.dense.weight] ({0}*{1} != {2})".format(
                n_config_relax, config.hidden_size, state_dict[_k].shape[0]))
            assert state_dict[_k].shape[0] % config.hidden_size == 0
            n_state_relax = int(state_dict[_k].shape[0]/config.hidden_size)
            assert (n_state_relax == 1) != (n_config_relax ==
                                            1), "!!!!n_state_relax == 1 xor n_config_relax == 1!!!!"
            if n_state_relax == 1:
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                    n_config_relax, 1, 1).reshape((n_config_relax*config.hidden_size, config.hidden_size))
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.unsqueeze(
                        0).repeat(n_config_relax, 1).view(-1)
            elif n_config_relax == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.view(
                    n_state_relax, config.hidden_size, config.hidden_size).select(0, _task_idx)
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.view(
                        n_state_relax, config.hidden_size).select(0, _task_idx)

        # initialize QKV
        _all_head_size = config.num_attention_heads * \
            int(config.hidden_size / config.num_attention_heads)
        n_config_num_qkv = 1 if (config.num_qkv < 1) else config.num_qkv
        for qkv_name in ('query', 'key', 'value'):
            _k = 'bert.encoder.layer.0.attention.self.{0}.weight'.format(
                qkv_name)
            if (_k in state_dict) and (n_config_num_qkv*_all_head_size != state_dict[_k].shape[0]):
                logger.info("n_config_num_qkv*_all_head_size != state_dict[_k] ({0}*{1} != {2})".format(
                    n_config_num_qkv, _all_head_size, state_dict[_k].shape[0]))
                for layer_idx in range(config.num_hidden_layers):
                    _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                        layer_idx, qkv_name)
                    assert state_dict[_k].shape[0] % _all_head_size == 0
                    n_state_qkv = int(state_dict[_k].shape[0]/_all_head_size)
                    assert (n_state_qkv == 1) != (n_config_num_qkv ==
                                                  1), "!!!!n_state_qkv == 1 xor n_config_num_qkv == 1!!!!"
                    if n_state_qkv == 1:
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                            n_config_num_qkv, 1, 1).reshape((n_config_num_qkv*_all_head_size, _all_head_size))
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.bias'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.unsqueeze(
                            0).repeat(n_config_num_qkv, 1).view(-1)
                    elif n_config_num_qkv == 1:
                        if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                            _task_idx = config.task_idx
                        else:
                            _task_idx = 0
                        assert _task_idx != 3, "[INVALID] _task_idx=3: n_config_num_qkv=1 (should be 2)"
                        if _task_idx == 0:
                            _qkv_idx = 0
                        else:
                            _qkv_idx = 1
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.view(
                            n_state_qkv, _all_head_size, _all_head_size).select(0, _qkv_idx)
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.bias'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.view(
                            n_state_qkv, _all_head_size).select(0, _qkv_idx)

        # 此处对词表中的不同位置的词向量进行替换 by sfeng;
        if _word_emb_map:
            _k = 'bert.embeddings.word_embeddings.weight'
            for _tgt, _src in _word_emb_map:
                state_dict[_k].data[_tgt, :].copy_(
                    state_dict[_k].data[_src, :])

        # 支持对bert词表进行扩增，添加源数据词表（论文中未使用该方法对词表进行扩增）
        if _source_vocab:
            _k = 'bert.embeddings.word_embeddings.weight'
            if len(state_dict[_k].data) < config.vocab_size:
                for _vocab in _source_vocab:
                    init_vocab_emb = torch.zeros(1, len(state_dict[_k].data[0]))
                    if state_dict[_k].data.is_cuda:
                        init_vocab_emb = init_vocab_emb.cuda()
                    if isnumber(_vocab):
                        for _vcb in _vocab:  # 逐字符词向量平均得到近似初始词向量
                            init_vocab_emb += state_dict[_k].data[_primary_vocabs.index(_vcb), :]
                        init_vocab_emb = init_vocab_emb / len(_vocab)  # mean pooling or sum pooling? mean pooling会出现一个问题：如9999的初始词向量与9一样；
                    else:
                        rand_num = random.sample([i for i in range(len(state_dict[_k].data))], 5)
                        for rand_n in rand_num:  # 随机初始化（多个向量取平均）
                            init_vocab_emb += state_dict[_k].data[rand_n, :]
                        init_vocab_emb = init_vocab_emb / len(rand_num)
                    state_dict[_k].data = torch.cat([state_dict[_k].data, init_vocab_emb], 0)

        # 加载bert预训练参数到模型中，替换掉原来的随机初始化的参数 by sfeng;
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):  # 完整的bert模型 add by sfeng
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def rescale_some_parameters(self):
        for layer_id, layer in enumerate(self.encoder.layer):
            layer.attention.output.dense.weight.data.div_(
                math.sqrt(2.0*(layer_id + 1)))
            layer.output.dense.weight.data.div_(math.sqrt(2.0*(layer_id + 1)))

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        try:
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except:
            pass
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, mask_qkv=None, task_idx=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, task_idx=task_idx)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class AnalogyModule(PreTrainedBertModel):  # 论文中的推理模块，构建两层的transformer结构 add by sfeng
    """Analogy Module for paper.
    refer to BertModel, the param and input most similar to BertModel.
    """

    def __init__(self, config):
        super(AnalogyModule, self).__init__(config)
        tmp_num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = 2  # 重定义推理模块层数
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)
        config.num_hidden_layers = tmp_num_hidden_layers  # 层数复原

    def rescale_some_parameters(self):
        for layer_id, layer in enumerate(self.encoder.layer):
            layer.attention.output.dense.weight.data.div_(
                math.sqrt(2.0*(layer_id + 1)))
            layer.output.dense.weight.data.div_(math.sqrt(2.0*(layer_id + 1)))

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        try:
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except:
            pass
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, analogy_input, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, mask_qkv=None, task_idx=None):
        extended_attention_mask = self.get_extended_attention_mask(
            analogy_input, token_type_ids, attention_mask)
        encoded_layers = self.encoder(analogy_input, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


class AnalogyModuleIncr(AnalogyModule):  # 继承自BertModel，跟BertModel相比在return部分多了个embedding_output，其他一样 add by sfeng
    def __init__(self, config):
        super(AnalogyModuleIncr, self).__init__(config)

    def forward(self, analogy_input, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None, mask_qkv=None, task_idx=None, is_analogy_input=False):
        extended_attention_mask = self.get_extended_attention_mask(
            analogy_input, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            analogy_input, token_type_ids, position_ids, task_idx=task_idx, is_analogy_input=is_analogy_input)

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids)
        sequence_output = encoded_layers[-1]
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


class BertModelIncr(BertModel):  # 继承自BertModel，跟BertModel相比在return部分多了个embedding_output，其他一样 add by sfeng
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, analogy_attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None, mask_qkv=None, task_idx=None, num_equ_ids=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)
        extended_analogy_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, analogy_attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, num_equ_ids, task_idx=task_idx)
        encoded_layers, attn_dists = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      extended_analogy_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output, attn_dists


class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, mask_qkv=None, task_idx=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))  # loss(x,class) = -log(exp(x[class]/sum_j exp(x[j]))) 其中，x为预测结果(各类别下的概率)，class为实际类别.
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertPreTrainingPairTransform(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingPairTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, pair_x, pair_y):
        hidden_states = torch.cat([pair_x, pair_y], dim=-1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainingPairRel(nn.Module):
    def __init__(self, config, num_rel=0):
        super(BertPreTrainingPairRel, self).__init__()
        self.R_xy = BertPreTrainingPairTransform(config)
        self.rel_emb = nn.Embedding(num_rel, config.hidden_size)

    def forward(self, pair_x, pair_y, pair_r, pair_pos_neg_mask):
        # (batch, num_pair, hidden)
        xy = self.R_xy(pair_x, pair_y)
        r = self.rel_emb(pair_r)
        _batch, _num_pair, _hidden = xy.size()
        pair_score = (xy * r).sum(-1)
        # torch.bmm(xy.view(-1, 1, _hidden),r.view(-1, _hidden, 1)).view(_batch, _num_pair)
        # .mul_(-1.0): objective to loss
        return F.logsigmoid(pair_score * pair_pos_neg_mask.type_as(pair_score)).mul_(-1.0)


class CopyNet2(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(CopyNet2, self).__init__()
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.vocab_size = config.vocab_size
        self.p_gen_linear = nn.Linear(config.hidden_size*2, 1, bias=False)

    def forward(self, x_e, dec_input, attn_dist, enc_batch_extend_vocab, extra_zeros):
        '''
        x_e:  shape [batch_size, top_k, pred_len, output_dim]
        dec_input:  shape [batch_size, top_k, pred_len, output_dim]
        attn_dist:  shape [batch_size, top_k, ques_seq_len, output_dim]
        enc_batch_extend_vocab:  shape [batch_size, top_k, ques_seq_len]
        extra_zeros:  shape [batch_size, top_k, equ_seq_len, vocab_size]
        '''
        p_gen_input = torch.cat((x_e, dec_input), 3)  # B x (2*hidden_dim)
        p_gen = self.p_gen_linear(p_gen_input)  #  [batch_size, top_k, pred_len, 1]
        p_gen = torch.sigmoid(p_gen)
        # gen_scores = self.phi_g(x_e)  # batch_size, top_k, pred_len, vocab_size
        gen_logit = self.decoder(x_e) + self.bias  # batch_size, top_k, pred_len, vocab_size
        vocab_dist = torch.softmax(gen_logit, dim=3)  
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dist = p_gen * vocab_dist
        attn_dist = (1-p_gen) * attn_dist
        if extra_zeros is not None:
            vocab_dist = torch.cat([vocab_dist, extra_zeros.float()], 3)
        final_dist = vocab_dist.scatter_add(3, enc_batch_extend_vocab, attn_dist)
        probs = final_dist.mean(1)  # batch_size, pred_len, vocab_size
        return torch.log(probs+1e-8)


class BertForPreTrainingLossMask(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, num_rel=0, num_sentlvl_labels=0, no_nsp=False, add_memory_module=False, max_len_a=192,
            max_len_b=64, add_copynet=False, vocabs=[], topk=2):
        super(BertForPreTrainingLossMask, self).__init__(config)
        # self.bert = BertModel(config)
        self.topk = topk
        self.vocabs = vocabs
        self.bert = BertModelIncr(config)
        self.copyNet = CopyNet2(config, self.bert.embeddings.word_embeddings.weight)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        # self.cls = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_sentlvl_labels = num_sentlvl_labels
        self.cls2 = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        if no_nsp:
            self.crit_next_sent = None
        else:
            self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.num_labels = num_labels
        self.num_rel = num_rel
        self.add_memory_module = add_memory_module
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.add_copynet = add_copynet
        self.config = config
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None
        self.apply(self.init_bert_weights)
        self.bert.rescale_some_parameters()
        self.config = config
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None, 
                masked_pos_2=None, masked_weights_2=None, masked_labels_2=None, mask_qkv=None, 
                analogy_attention_mask=None, step=0, position_ids=None, num_equ_ids=None,
                extend_oov_query_qsids=None, extra_zeros=None):

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        embedding_output, sequence_output, pooled_output, attn_dists = self.bert(input_ids, token_type_ids, position_ids, attention_mask, analogy_attention_mask, 
                        output_all_encoded_layers=True, mask_qkv=mask_qkv, task_idx=task_idx, num_equ_ids=num_equ_ids)
        sequence_output = sequence_output[-1]

        if self.add_copynet:
            question_ids = input_ids.view(-1, self.topk, input_ids.size(1))
            batch_size, ques_seq_len, equ_seq_len, vocab_size = question_ids.size(0), self.max_len_a, self.max_len_b, self.config.vocab_size
            split_list = [ques_seq_len, equ_seq_len, ques_seq_len, equ_seq_len]
            sequence_output_final = sequence_output.view(-1, self.topk, sequence_output.size(1), sequence_output.size(2))
            emb_output = embedding_output.view(-1, self.topk, embedding_output.size(1), embedding_output.size(2))   
            _, _, _, dec_input = torch.split(emb_output, split_list, dim=2)   
            _, _, _, x_e = torch.split(sequence_output_final, split_list, dim=2)      
            attn_dist = attn_dists[-1].view(-1, self.topk, attn_dists[-1].size(1), attn_dists[-1].size(2))
            # 提取query question对应的注意力，并对attn_dist进行归一化处理
            row_start = ques_seq_len + equ_seq_len + ques_seq_len
            column_start, column_end = ques_seq_len + equ_seq_len, row_start 
            attn_dist_tmp = attn_dist[:, :, row_start:, column_start: column_end]
            normalization_factor = attn_dist_tmp.sum(3, keepdim=True)
            attn_dist = attn_dist_tmp / (normalization_factor + 1e-8)

            extend_oov_query_qsids = extend_oov_query_qsids.view(-1, self.topk, extend_oov_query_qsids.size(1))
            extend_oov_query_qsids = extend_oov_query_qsids.unsqueeze(2).expand(-1, -1, self.max_len_b, -1)
            if extra_zeros is not None:
                extra_zeros = extra_zeros.view(-1, self.topk, extra_zeros.size(1))
                extra_zeros = extra_zeros.unsqueeze(2).expand(-1, -1, self.max_len_b, -1)
            log_scores = self.copyNet(x_e, dec_input, attn_dist, extend_oov_query_qsids, extra_zeros)

            masked_labels_2 = torch.index_select(masked_labels_2.view(batch_size, self.topk, -1), 1, torch.tensor([0]).cuda()).squeeze(1)
            masked_weights_2 = torch.index_select(masked_weights_2.view(batch_size, self.topk, -1), 1, torch.tensor([0]).cuda()).squeeze(1)
            copy_crit_mask_lm_smoothed = LabelSmoothingLoss(
                        self.config.label_smoothing, log_scores.size(2), ignore_index=0, reduction='none').cuda()
            copy_loss = copy_crit_mask_lm_smoothed(log_scores, masked_labels_2)
            copy_loss = loss_mask_and_normalize(copy_loss.float(), masked_weights_2)
        
        if masked_lm_labels is None:
            if masked_pos is None:
                prediction_scores, seq_relationship_score = self.cls(
                    sequence_output, pooled_output, task_idx=task_idx)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
            return prediction_scores, seq_relationship_score

        # masked lm
        sequence_output_masked = gather_seq_out_by_pos(
            sequence_output, masked_pos)
        prediction_scores_masked, seq_relationship_score = self.cls(
            sequence_output_masked, pooled_output, task_idx=task_idx)
        if self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
        else:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
        masked_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weights)

        if self.add_copynet:
            if step > 0 and step % 2000 == 0:
                logging.info('masked_lm_loss: {:.4f}, copy_loss: {:.4f}'.format(masked_lm_loss, copy_loss))
            masked_lm_loss = masked_lm_loss + copy_loss
        elif self.add_memory_module:
            # memory module loss
            if self.cls2 is not None and masked_pos_2 is not None:
                sequence_output_masked_2 = gather_seq_out_by_pos(
                    sequence_output, masked_pos_2)
                prediction_scores_masked_2, _ = self.cls2(sequence_output_masked_2, None)
                masked_lm_loss_2 = self.crit_mask_lm(
                    prediction_scores_masked_2.transpose(1, 2).float(), masked_labels_2)
                masked_lm_loss_2 = loss_mask_and_normalize(
                    masked_lm_loss_2.float(), masked_weights_2)

                if step % 2000 == 0:
                    logging.info('masked_lm_loss: {:.4f}, masked_lm_loss_2: {:.4f}'.format(masked_lm_loss, masked_lm_loss_2))
                masked_lm_loss = masked_lm_loss + masked_lm_loss_2

        return masked_lm_loss


class BertForSeq2SeqDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, mask_word_id=0, num_labels=2, num_rel=0, search_beam_size=1, length_penalty=1.0, 
                eos_id=0, sos_id=0, forbid_duplicate_ngrams=False, forbid_ignore_set=None, not_predict_set=None, 
                ngram_size=3, min_len=0, mode="s2s", pos_shift=False, add_memory_module=False, topk=2, max_len_a=192, 
                max_len_b=64, add_copynet=False, vocabs=[]):
        super(BertForSeq2SeqDecoder, self).__init__(config)
        self.vocabs = vocabs
        self.bert = BertModelIncr(config)
        # self.pri_bert = BertModel(config)
        self.copyNet = CopyNet2(config, self.bert.embeddings.word_embeddings.weight)

        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.num_rel = num_rel
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.not_predict_set = not_predict_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pos_shift = pos_shift
        self.add_memory_module = add_memory_module
        self.topk = topk
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.add_copynet = add_copynet       

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None, 
            analogy_attention_mask=None, retrieve_batch=None, num_equ_ids=None, extend_oov_query_qsids=[], extra_zeros=[]):
        if self.search_beam_size > 1:
            return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask, task_idx=task_idx, mask_qkv=mask_qkv, 
                analogy_attention_mask=analogy_attention_mask, retrieve_batch=retrieve_batch, num_equ_ids=num_equ_ids, 
                extend_oov_query_qsids=extend_oov_query_qsids, extra_zeros=extra_zeros)

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        if self.pos_shift:
            sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)

        if self.add_copynet:
            question_ids = input_ids.view(-1, self.topk, input_ids.size(1))
            batch_size, ques_seq_len, equ_seq_len, vocab_size = question_ids.size(0), self.max_len_a, self.max_len_b, self.config.vocab_size
            extend_oov_query_qsids = extend_oov_query_qsids.view(-1, self.topk, extend_oov_query_qsids.size(1))
            extend_oov_query_qsids = extend_oov_query_qsids.unsqueeze(2).expand(-1, -1, 1, -1)
            if extra_zeros is not None:
                extra_zeros = extra_zeros.view(-1, self.topk, extra_zeros.size(1))
                extra_zeros = extra_zeros.unsqueeze(2).expand(-1, -1, 1, -1)

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            if self.pos_shift:
                if next_pos == input_length:
                    x_input_ids = torch.cat((curr_ids, sos_ids), dim=1)
                    start_pos = 0
                else:
                    x_input_ids = curr_ids
                    start_pos = next_pos
            else:
                start_pos = next_pos - curr_length
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)  # 添加一个mask标志，下文encode部分就是为了预测这个mask token

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_analogy_attention_mask = analogy_attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            if num_equ_ids is not None:
                curr_num_equ_ids = num_equ_ids[:, start_pos:next_pos + 1] 
            else:
                curr_num_equ_ids = None
            new_embedding, new_encoded_layers, _, attn_dists = \
                self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask, curr_analogy_attention_mask, output_all_encoded_layers=True, 
                          prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, num_equ_ids=curr_num_equ_ids)
            
            last_hidden = new_encoded_layers[-1][:, -1:, :]

            if self.add_copynet:
                last_hidden = new_encoded_layers[-1]
                row_start = ques_seq_len + equ_seq_len + ques_seq_len
                column_start, column_end = ques_seq_len + equ_seq_len, row_start 
                split_list = [ques_seq_len, equ_seq_len, ques_seq_len, last_hidden.size(1)-ques_seq_len*2-equ_seq_len]
                if start_pos == 0:
                    last_hidden = last_hidden.view(-1, self.topk, last_hidden.size(1), last_hidden.size(2))
                    _, _, _, x_e = torch.split(last_hidden, split_list, dim=2)      
                    emb_output = new_embedding.view(-1, self.topk, new_embedding.size(1), new_embedding.size(2))   
                    _, _, _, dec_input = torch.split(emb_output, split_list, dim=2)   
                    attn_dist = attn_dists[-1].view(-1, self.topk, attn_dists[-1].size(1), attn_dists[-1].size(2))
                    # 如果只对query question进行copy，需要对attn_dist进行归一化处理
                    attn_dist = attn_dist[:, :, row_start:, column_start: column_end]
                    normalization_factor = attn_dist.sum(3, keepdim=True)
                    attn_dist = attn_dist / (normalization_factor + 1e-8)
                else:
                    seq_len, token_dim = last_hidden.size(1), last_hidden.size(2)
                    last_hidden = last_hidden.view(batch_size, self.topk, seq_len, token_dim)
                    x_e = last_hidden[:, :, -1:, :]
                    dec_input = new_embedding.view(batch_size, self.topk, seq_len, token_dim)
                    attn_row, attn_column = attn_dists[-1].size(1), attn_dists[-1].size(2)
                    attn_dist = attn_dists[-1].view(batch_size, self.topk, attn_row, attn_column)
                    attn_dist = attn_dist[:, :, -1:, column_start: column_end]
                    normalization_factor = attn_dist.sum(3, keepdim=True)
                    attn_dist = attn_dist / (normalization_factor + 1e-8)
                log_scores = self.copyNet(x_e, dec_input, attn_dist, extend_oov_query_qsids, extra_zeros)

            elif self.add_memory_module:
                # [调和平均的话，这里应该是topk个结果平均] 
                prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx)
                probs = torch.softmax(prediction_scores, dim=-1)
                probs = probs.view(-1, self.topk, probs.size(1), probs.size(2)).mean(1)
                log_scores = torch.log(probs+1e-8)
            else:
                prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx)
                log_scores = torch.nn.functional.log_softmax(prediction_scores, dim=-1)

            _, max_ids = torch.max(log_scores, dim=-1)  # 取出概率最大的id
            output_ids.append(max_ids)

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = new_embedding
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
            else:
                if prev_embedding is None:
                    prev_embedding = new_embedding[:, :-1, :]  # 不取添加的mask_ids对应的embedding向量；
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x[:, :-1, :]
                                           for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
            
            if self.add_memory_module:
                if not self.add_copynet and start_pos == 0:
                    batch_size = int(batch_size / self.topk)
                tmp_max_ids = max_ids.unsqueeze(1).expand(-1,self.topk,-1).contiguous().view(-1,max_ids.size(1))
                curr_ids = torch.reshape(tmp_max_ids, [batch_size * self.topk, 1])
            else:
                curr_ids = max_ids

            # 对于解码出oov字符的情况，在作为下一步的输入时将其转换为unk;
            if curr_ids.max() >= len(self.vocabs):
                curr_ids = torch.reshape(curr_ids, [-1,])
                curr_ids = torch.tensor([id if id < len(self.vocabs) else self.vocabs.index('[UNK]') for id in curr_ids]).cuda()
                curr_ids = torch.reshape(curr_ids, [-1, 1])

            next_pos += 1
        
        return torch.cat(output_ids, dim=1)


    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None,
            analogy_attention_mask=None, retrieve_batch=None, num_equ_ids=None, extend_oov_query_qsids=[], extra_zeros=[]):

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        if self.pos_shift:
            sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None
        prev_analogy_layers = None

        if self.add_copynet:
            question_ids = input_ids.view(-1, self.topk, input_ids.size(1))
            batch_size, ques_seq_len, equ_seq_len, vocab_size = question_ids.size(0), self.max_len_a, self.max_len_b, self.config.vocab_size
            extend_oov_query_qsids = extend_oov_query_qsids.view(-1, self.topk, extend_oov_query_qsids.size(1))
            extend_oov_query_qsids = extend_oov_query_qsids.unsqueeze(2).expand(-1, -1, 1, -1)
            if extra_zeros is not None:
                extra_zeros = extra_zeros.view(-1, self.topk, extra_zeros.size(1))
                extra_zeros = extra_zeros.unsqueeze(2).expand(-1, -1, 1, -1)

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            if self.pos_shift:
                if next_pos == input_length:
                    x_input_ids = torch.cat((curr_ids, sos_ids), dim=1)
                    start_pos = 0
                else:
                    x_input_ids = curr_ids
                    start_pos = next_pos
            else:
                start_pos = next_pos - curr_length
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_analogy_attention_mask = analogy_attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            if num_equ_ids is not None:
                curr_num_equ_ids = num_equ_ids[:, start_pos:next_pos + 1] 
            else:
                curr_num_equ_ids = None

            new_embedding, new_encoded_layers, _, attn_dists = \
                self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask, curr_analogy_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers, 
                          mask_qkv=mask_qkv, num_equ_ids=curr_num_equ_ids)
            last_hidden = new_encoded_layers[-1][:, -1:, :]

            if self.add_copynet:
                last_hidden = new_encoded_layers[-1]
                row_start = ques_seq_len + equ_seq_len + ques_seq_len
                column_start, column_end = ques_seq_len + equ_seq_len, row_start 
                split_list = [ques_seq_len, equ_seq_len, ques_seq_len, last_hidden.size(1)-ques_seq_len*2-equ_seq_len]
                if start_pos == 0:
                    last_hidden = last_hidden.view(-1, self.topk, last_hidden.size(1), last_hidden.size(2))
                    _, _, _, x_e = torch.split(last_hidden, split_list, dim=2)      
                    emb_output = new_embedding.view(-1, self.topk, new_embedding.size(1), new_embedding.size(2))   
                    _, _, _, dec_input = torch.split(emb_output, split_list, dim=2)   
                    attn_dist = attn_dists[-1].view(-1, self.topk, attn_dists[-1].size(1), attn_dists[-1].size(2))
                    # 如果只对query question进行copy，需要对attn_dist进行归一化处理
                    attn_dist = attn_dist[:, :, row_start:, column_start: column_end]
                    normalization_factor = attn_dist.sum(3, keepdim=True)
                    attn_dist = attn_dist / (normalization_factor + 1e-8)
                else:
                    seq_len, token_dim = last_hidden.size(1), last_hidden.size(2)
                    last_hidden = last_hidden.view(batch_size, self.topk, K, seq_len, token_dim).permute(0,2,1,3,4)
                    last_hidden = last_hidden.contiguous().view(-1, self.topk, seq_len, token_dim)
                    x_e = last_hidden[:, :, -1:, :]

                    emb_output = new_embedding.view(batch_size, self.topk, K, seq_len, token_dim).permute(0,2,1,3,4)
                    dec_input = emb_output.contiguous().view(-1, self.topk, seq_len, token_dim)
                    attn_row, attn_column = attn_dists[-1].size(1), attn_dists[-1].size(2)
                    attn_dist = attn_dists[-1].view(batch_size, self.topk, K, attn_row, attn_column).permute(0,2,1,3,4)
                    attn_dist = attn_dist.contiguous().view(-1, self.topk, attn_row, attn_column)
                    attn_dist = attn_dist[:, :, -1:, column_start: column_end]
                    normalization_factor = attn_dist.sum(3, keepdim=True)
                    attn_dist = attn_dist / (normalization_factor + 1e-8)
                log_scores = self.copyNet(x_e, dec_input, attn_dist, extend_oov_query_qsids, extra_zeros)

            elif self.add_memory_module:
                # [TODO:调和平均的话，这里应该是topk个结果平均] 
                if start_pos == 0:
                    batch_size = int(batch_size / self.topk)
                else:
                    seq_len, token_dim = last_hidden.size(1), last_hidden.size(2)
                    last_hidden = last_hidden.view(batch_size, self.topk, K, seq_len, token_dim).permute(0,2,1,3,4)
                    last_hidden = last_hidden.contiguous().view(-1, seq_len, token_dim)

                prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx)
                probs = torch.softmax(prediction_scores, dim=-1)
                probs = probs.view(-1, self.topk, probs.size(1), probs.size(2)).mean(1)
                log_scores = torch.log(probs+1e-8)
            else:
                prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx)
                log_scores = torch.nn.functional.log_softmax(prediction_scores, dim=-1)

            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)  # 对ngram_size下重复的字符进行mask，即下一个字符不要预测成该重复字符（在解题场景下不适合）
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            if self.not_predict_set:
                for token_id in self.not_predict_set:
                    log_scores[:, :, token_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)  
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)  # 从kk_ids中K*K个候选词中选出topK个；
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                if self.add_memory_module:
                    x = torch.reshape(x, [batch_size*self.topk, K] + x_shape[1:])
                else:
                    x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding)
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                    if self.add_memory_module:
                        tmp_back_ptrs = back_ptrs.unsqueeze(1).expand(-1,self.topk,-1).contiguous().view(-1,back_ptrs.size(1)) ## 修改back_ptrs维度
                        prev_embedding = select_beam_items(prev_embedding, tmp_back_ptrs)
                    else:
                        prev_embedding = select_beam_items(prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
                    if self.add_memory_module:
                        tmp_back_ptrs = back_ptrs.unsqueeze(1).expand(-1,self.topk,-1).contiguous().view(-1,back_ptrs.size(1)) ## 修改back_ptrs维度
                        prev_encoded_layers = [select_beam_items(x, tmp_back_ptrs) for x in prev_encoded_layers]
                    else:
                        prev_encoded_layers = [select_beam_items(
                            x, back_ptrs) for x in prev_encoded_layers]
            else:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding[:, :-1, :])
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x[:, :-1, :]) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]

            if self.add_memory_module:
                tmp_k_ids = k_ids.unsqueeze(1).expand(-1,self.topk,-1).contiguous().view(-1,k_ids.size(1))
                curr_ids = torch.reshape(tmp_k_ids, [batch_size * self.topk * K, 1])
            else:
                curr_ids = torch.reshape(k_ids, [batch_size * K, 1])
            if curr_ids.max() >= len(self.vocabs):
                curr_ids = torch.reshape(curr_ids, [-1,])
                curr_ids = torch.tensor([id if id < len(self.vocabs) else self.vocabs.index('[UNK]') for id in curr_ids]).cuda()
                curr_ids = torch.reshape(curr_ids, [-1, 1])
            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)
                if mask_qkv is not None:
                    mask_qkv = first_expand(mask_qkv)
                if num_equ_ids is not None:
                    num_equ_ids = first_expand(num_equ_ids)
                
                # 扩展检索question为beam_size倍； add by sfeng;
                analogy_attention_mask = first_expand(analogy_attention_mask)
                if self.add_copynet:
                    extend_oov_query_qsids = first_expand(extend_oov_query_qsids)
                    if extra_zeros is not None:
                        extra_zeros = first_expand(extra_zeros)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces


class BertForMaskedLM(PreTrainedBertModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, mask_qkv=None, task_idx=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(PreTrainedBertModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None, mask_qkv=None, task_idx=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, mask_qkv=None, task_idx=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if labels.dtype == torch.long:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif labels.dtype == torch.half or labels.dtype == torch.float:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                print('unkown labels.dtype')
                loss = None
            return loss
        else:
            return logits


class BertForMultipleChoice(PreTrainedBertModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, mask_qkv=None, task_idx=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(
            flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(PreTrainedBertModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, mask_qkv=None, task_idx=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: either
            - a BertConfig class instance with the configuration to build a new model, or
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-base-multilingual`
                . `bert-base-chinese`
                The pre-trained model will be downloaded and cached if needed.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, task_idx=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, task_idx=task_idx)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
