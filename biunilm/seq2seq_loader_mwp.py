from random import randint, shuffle, choice
import math
import torch
import pickle

from biunilm.loader_utils import batch_list_to_batch_tensors, Pipeline, raw_batch_list_to_batch_tensors
from utils.tool import isnumber


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, memory_file, batch_size, tokenizer, is_single_char=False, add_memory_module=False, 
            bi_uni_pipeline=[], topk=2, is_train=None):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.is_single_char = is_single_char
        self.add_memory_module = add_memory_module
        self.ex_list = []
        self.topk = topk
        self.index_ids = {}
        self.is_train = is_train

        question_sim_qsas = pickle.load(open(memory_file, 'rb'))
        for idx, (id, value) in enumerate(question_sim_qsas.items()):
            sim_ex_list = []
            primary_question, sim_question = value

            if self.is_train:
                pri_src, pri_tgt = self.tokenize_input(primary_question[0], primary_question[1])
            else:
                pri_src = self.tokenizer.tokenize(primary_question[0].lower().strip())
                pri_tgt = primary_question[1]

            pri_src_keep_unk = self.tokenizer.tokenize_sf(primary_question[0].lower().strip())
            if self.is_train:
                pri_tgt_tk = ' '.join(eval(primary_question[1].lower().strip()))
                pri_tgt_keep_unk = self.tokenizer.tokenize_sf(pri_tgt_tk)
            else:
                pri_tgt_keep_unk = None

            pri_src = [pri_src, pri_src_keep_unk]
            pri_tgt = [pri_tgt, pri_tgt_keep_unk]

            sub_ex_list = [pri_src, pri_tgt]

            for i, (question, equation, _) in enumerate(sim_question):
                if i >= self.topk:  # 取topk个相似题进行计算
                    break
                sim_src, sim_tgt = self.tokenize_input(question, equation)
                sim_ex_list.extend([sim_src, sim_tgt])
            sub_ex_list.extend(sim_ex_list)
            self.ex_list.append(sub_ex_list)
            self.index_ids[idx] = id

        print('Load {0} documents'.format(len(self.ex_list)))

    def tokenize_input(self, src, tgt):
        src_tk = self.tokenizer.tokenize(src.lower().strip())
        pri_tgt_tk = ' '.join(eval(tgt.lower().strip()))
        tgt_tk = self.tokenizer.tokenize(pri_tgt_tk)
        return src_tk, tgt_tk

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instances = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        new_instance = []
        query_question = instances[:2]
        for i in range(2, len(instances), 2):
            pro_instance = proc(query_question + instances[i:i+2])
            new_instance.append(pro_instance)
        return new_instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, vocab_words, indexer, max_len=512, truncate_config={}, 
            max_analogy_len=512, is_train=False, copynet_name='copynet1'):
        super().__init__()
        self.max_pred = max_pred  # max tokens of prediction
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_analogy_len, max_analogy_len), dtype=torch.long))
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.max_analogy_len = max_analogy_len
        self.is_train = is_train
        self.copynet_name = copynet_name

    def process_tokens(self, tokens_a_wo_padding, tokens_b):
        tokens_a_wo_padding = tokens_a_wo_padding[:self.max_len_a - 2]
        tokens_a = ['[CLS]'] + tokens_a_wo_padding + ['[SEP]']
        if len(tokens_a) < self.max_len_a:
            tokens_a += ['[PAD]'] * (self.max_len_a - len(tokens_a))  
        if tokens_b is not None:
            tokens_b = tokens_b[:self.max_len_b-1]
            tokens = tokens_a + tokens_b + ['[SEP]'] + ['[PAD]'] * (self.max_len_b - len(tokens_b)-1) 
        else:
            tokens = tokens_a
        return tokens_a_wo_padding, tokens_a, tokens_b, tokens

    def gain_num_equ_ids(self, token):
        num_equ_ids = []
        for tok in token:
            if isnumber(tok) or tok.startswith('##') or tok == '.':
                num_equ_ids.append(1)
            elif tok in ['+', '-', '*', '/', '^']:
                num_equ_ids.append(2)
            else:
                num_equ_ids.append(0)
        return num_equ_ids

    def add_question_oov_ids(self, tokens):
        """添加question oov ids."""
        oovs = []
        extend_oov_token_ids = []
        unk_id = self.indexer(['[UNK]'])[0]
        token_ids = self.indexer(tokens)
        for i, id in enumerate(token_ids):
            if id == unk_id:
                # if id not in oovs:
                if tokens[i] not in oovs:
                    oovs.append(tokens[i])
                oov_num = oovs.index(tokens[i])
                extend_oov_token_ids.append(len(self.vocab_words)+oov_num)
            else:
                extend_oov_token_ids.append(id)
        return extend_oov_token_ids, oovs

    def add_equation_oov_ids(self, tokens, oovs):
        """添加equation oov ids."""
        extend_oov_token_ids = []
        unk_id = self.indexer(['[UNK]'])[0]
        token_ids = self.indexer(tokens)
        for i, id in enumerate(token_ids):
            if id == unk_id:
                if tokens[i] in oovs:
                    oov_num = oovs.index(tokens[i])
                    extend_oov_token_ids.append(len(self.vocab_words)+oov_num)
                else:
                    extend_oov_token_ids.append(unk_id)
            else:
                extend_oov_token_ids.append(id)
        return extend_oov_token_ids

    def __call__(self, instance):
        query_question, query_equation, ret_tokens_a_wo_padding, ret_tokens_b = instance  # ret=retrieve
        query_tokens_a_wo_padding, query_tokens_a_wo_padding_keep_unk = query_question
        query_tokens_b, query_tokens_b_keep_unk = query_equation
        # add start sos for position shift.
        ret_tokens_b = ['[S2S_SOS]'] + ret_tokens_b
        if query_tokens_b is not None:
            query_tokens_b = ['[S2S_SOS]'] + query_tokens_b

        query_tokens_a_wo_padding, query_tokens_a, query_tokens_b, query_tokens = self.process_tokens(query_tokens_a_wo_padding, query_tokens_b)
        ret_tokens_a_wo_padding, ret_tokens_a, ret_tokens_b, ret_tokens = self.process_tokens(ret_tokens_a_wo_padding, ret_tokens_b)

        # add segment_ids
        segment_ids = [0]*self.max_len_a + [1]*self.max_len_b + [2]*self.max_len_a + [3]*self.max_len_b 

        # add position_ids, when add tokens_a_padding, 原来模型中的position_ids没有跳过padding部分，所以此处做修改；
        position_ids = [i for i in range(len(ret_tokens_a_wo_padding) + 2)]
        max_mark = position_ids[-1]
        position_ids += [0] * (self.max_len_a - len(ret_tokens_a_wo_padding) - 2)
        position_ids += [max_mark + 1 + i for i in range(len(ret_tokens_b)+1)]
        max_mark = position_ids[-1]
        position_ids += [0] * (self.max_len_b - len(ret_tokens_b) - 1)

        position_ids += [max_mark + 1 + i for i in range(len(query_tokens_a_wo_padding) + 2)]
        max_mark = position_ids[-1]
        position_ids += [0] * (self.max_len_a - len(query_tokens_a_wo_padding) - 2)
        if self.is_train:
            position_ids += [max_mark + 1 + i for i in range(len(query_tokens_b) + 1)]
            max_mark = position_ids[-1]
            position_ids += [0] * (self.max_len_b - len(query_tokens_b) - 1)
        else:
            position_ids += [max_mark + 1 + i for i in range(self.max_len_b)]

        # add num embedding.
        num_equ_ids = self.gain_num_equ_ids(ret_tokens)
        num_equ_ids += self.gain_num_equ_ids(query_tokens)
        num_equ_ids += [0] * (self.max_analogy_len-len(num_equ_ids))

        # add question and equation oov ids.
        _, query_tokens_a_keep_unk, _, _ = self.process_tokens(query_tokens_a_wo_padding_keep_unk, None)
        extend_oov_query_qsids, oov_tokens = self.add_question_oov_ids(query_tokens_a_keep_unk)

        # add pos, labels, weights, only used when training.
        if self.is_train:
            query_n_pred = min(self.max_pred, len(query_tokens_b))
            query_masked_pos = [self.max_len+self.max_len_a+i for i in range(len(query_tokens_b[:query_n_pred]))]
            query_masked_weights = [1]*query_n_pred
            query_masked_ids = self.indexer(query_tokens_b[1:query_n_pred]+['[SEP]'])
            if self.copynet_name == 'copynet2':
                query_keep_unk_len = len(query_tokens_b_keep_unk) if len(query_tokens_b_keep_unk) < self.max_pred else self.max_pred-1
                query_masked_ids = self.add_equation_oov_ids(query_tokens_b_keep_unk[:query_keep_unk_len]+['[SEP]'], oov_tokens)
            # Zero Padding for masked target
            if self.max_pred > query_n_pred:
                n_pad = self.max_pred - query_n_pred
                if query_masked_pos is not None:
                    query_masked_pos.extend([0]*n_pad)
                if query_masked_weights is not None:
                    query_masked_weights.extend([0]*n_pad)
                    # query_masked_weights.extend([1]*n_pad)  # 让所有位置的字符都进行loss计算（包括pad)
                if query_masked_ids is not None:
                    # query_masked_ids.extend([0]*n_pad)
                    query_masked_ids.extend([0]*(self.max_pred-len(query_masked_ids)))

            ret_n_pred = min(self.max_pred, len(ret_tokens_b))
            ret_masked_pos = [self.max_len_a+i for i in range(len(ret_tokens_b[:ret_n_pred]))]
            ret_masked_weights = [1]*ret_n_pred
            ret_masked_ids = self.indexer(ret_tokens_b[1:ret_n_pred]+['[SEP]'])
            # Zero Padding for masked target
            if self.max_pred > ret_n_pred:
                n_pad = self.max_pred - ret_n_pred
                if ret_masked_pos is not None:
                    ret_masked_pos.extend([0]*n_pad)
                if ret_masked_weights is not None:
                    ret_masked_weights.extend([0]*n_pad)
                    # ret_masked_weights.extend([1]*n_pad)  # 让所有位置的字符都进行loss计算（包括pad)
                if ret_masked_ids is not None:
                    ret_masked_ids.extend([0]*n_pad)
            
        # Token Indexing
        tokens = ret_tokens + query_tokens
        input_ids = self.indexer(tokens)

        # Zero Padding
        if self.is_train:
            if len(input_ids) > self.max_analogy_len:
                input_ids = input_ids[:self.max_analogy_len]
            else:
                n_pad = self.max_analogy_len - len(input_ids)
                input_ids.extend([0]*n_pad)

        mask_qkv = None

        # input mask
        input_mask = torch.zeros(self.max_analogy_len, self.max_analogy_len, dtype=torch.long)
        input_mask[:self.max_len, :len(ret_tokens_a_wo_padding)+2].fill_(1)
        second_st, second_end = self.max_len_a, self.max_len_a+len(ret_tokens_b)+1
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])
        input_mask[self.max_len:, self.max_len:self.max_len+len(query_tokens_a_wo_padding)+2].fill_(1)
        if self.is_train:
            fourth_st, fourth_end = self.max_len+self.max_len_a, self.max_len+self.max_len_a+len(query_tokens_b)+1
        else:
            fourth_st, fourth_end = self.max_len+self.max_len_a, self.max_analogy_len
        input_mask[fourth_st:fourth_end, fourth_st:fourth_end].copy_(
            self._tril_matrix[:fourth_end-fourth_st, :fourth_end-fourth_st])
        
        # analogy module mask
        analogy_attention_mask = torch.zeros(self.max_analogy_len, self.max_analogy_len, dtype=torch.long)
        analogy_attention_mask[:, :len(ret_tokens_a_wo_padding)+2].fill_(1)
        second_st, second_end = self.max_len_a, self.max_len_a+len(ret_tokens_b)+1
        analogy_attention_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])
        analogy_attention_mask[self.max_len:, self.max_len_a:self.max_len_a+len(ret_tokens_b)+1].fill_(1)
        analogy_attention_mask[self.max_len:, :self.max_len+len(query_tokens_a_wo_padding)+2].fill_(1)
        if self.is_train:
            fifth_st, fifth_end = self.max_len+self.max_len_a, self.max_len+self.max_len_a+len(query_tokens_b)+1
        else:
            fifth_st, fifth_end = self.max_len+self.max_len_a, self.max_analogy_len
        analogy_attention_mask[fifth_st:fifth_end, fifth_st:fifth_end].copy_(
            self._tril_matrix[:fifth_end-fifth_st, :fifth_end-fifth_st])

        if self.is_train:
            return [input_ids, segment_ids, input_mask, mask_qkv, query_masked_ids, query_masked_pos, query_masked_weights, ret_masked_ids, 
                    ret_masked_pos, ret_masked_weights, -1, self.task_idx, analogy_attention_mask, position_ids, num_equ_ids, extend_oov_query_qsids, oov_tokens]
        else:
            return [input_ids, segment_ids, input_mask, mask_qkv, self.task_idx, analogy_attention_mask, position_ids, num_equ_ids, extend_oov_query_qsids, oov_tokens]


class Preprocess4Seq2seqBaseline(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, vocab_words, indexer, max_len=512, truncate_config={}, 
            max_analogy_len=512, is_train=False, copynet_name='copynet1'):
        super().__init__()
        self.max_pred = max_pred  # max tokens of prediction
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.is_train = is_train

    def process_tokens(self, tokens_a_wo_padding, tokens_b):
        tokens_a_wo_padding = tokens_a_wo_padding[:self.max_len_a - 2]
        tokens_a = ['[CLS]'] + tokens_a_wo_padding + ['[SEP]']
        if len(tokens_a) < self.max_len_a:
            tokens_a += ['[PAD]'] * (self.max_len_a - len(tokens_a))  
        if tokens_b is not None:
            tokens_b = tokens_b[:self.max_len_b-1]
            tokens = tokens_a + tokens_b + ['[SEP]'] + ['[PAD]'] * (self.max_len_b - len(tokens_b)-1) 
        else:
            tokens = tokens_a
        return tokens_a_wo_padding, tokens_a, tokens_b, tokens

    def __call__(self, instance):
        query_question, query_equation, ret_tokens_a_wo_padding, ret_tokens_b = instance  # ret=retrieve
        query_tokens_a_wo_padding, query_tokens_a_wo_padding_keep_unk = query_question
        query_tokens_b, query_tokens_b_keep_unk = query_equation
        # add start sos for position shift.
        ret_tokens_b = ['[S2S_SOS]'] + ret_tokens_b
        if query_tokens_b is not None:
            query_tokens_b = ['[S2S_SOS]'] + query_tokens_b

        query_tokens_a_wo_padding, query_tokens_a, query_tokens_b, query_tokens = self.process_tokens(query_tokens_a_wo_padding, query_tokens_b)

        # add segment_ids
        if self.is_train:
            segment_ids = [0]*self.max_len_a + [1]*(len(query_tokens_b)+1)
            segment_ids += [0]*(self.max_len-len(segment_ids))
        else:
            segment_ids = [0]*self.max_len_a + [1]*self.max_len_b

        # add position_ids, when add tokens_a_padding, 原来模型中的position_ids没有跳过padding部分，所以此处做修改；
        position_ids = [i for i in range(len(query_tokens_a_wo_padding) + 2)]
        max_mark = position_ids[-1]
        position_ids += [0] * (self.max_len_a - len(query_tokens_a_wo_padding) - 2)
        if self.is_train:
            position_ids += [max_mark + 1 + i for i in range(len(query_tokens_b) + 1)]
            position_ids += [0] * (self.max_len_b - len(query_tokens_b) - 1)
        else:
            max_mark = position_ids[-1]
            position_ids += [max_mark + 1 + i for i in range(self.max_len_b)]

        # add pos, labels, weights, only used when training.
        if self.is_train:
            query_n_pred = min(self.max_pred, len(query_tokens_b))
            query_masked_pos = [self.max_len_a+i for i in range(len(query_tokens_b[:query_n_pred]))]
            query_masked_weights = [1]*query_n_pred
            query_masked_ids = self.indexer(query_tokens_b[1:query_n_pred]+['[SEP]'])
            # Zero Padding for masked target
            if self.max_pred > query_n_pred:
                n_pad = self.max_pred - query_n_pred
                if query_masked_pos is not None:
                    query_masked_pos.extend([0]*n_pad)
                if query_masked_weights is not None:
                    query_masked_weights.extend([0]*n_pad)
                if query_masked_ids is not None:
                    query_masked_ids.extend([0]*n_pad)
            
        # Token Indexing
        input_ids = self.indexer(query_tokens)

        # Zero Padding
        if self.is_train:
            if len(input_ids) > self.max_len:
                input_ids = input_ids[:self.max_len]
            else:
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0]*n_pad)

        mask_qkv = None

        # input mask.
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(query_tokens_a_wo_padding)+2].fill_(1)
        # first_end = len(query_tokens_a_wo_padding)+2
        # input_mask[:first_end, :first_end].fill_(1)
        if self.is_train:
            second_st, second_end = self.max_len_a, self.max_len_a+len(query_tokens_b)+1
        else:
            second_st, second_end = self.max_len_a, self.max_len
        # input_mask[second_st:second_end, :first_end].fill_(1)
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])
        
        # analogy module mask.
        analogy_attention_mask = input_mask

        # fill None.
        oov_tokens = []
        if self.is_train:
            ret_masked_ids = query_masked_ids
            ret_masked_pos = query_masked_pos
            ret_masked_weights = query_masked_weights
        else:
            ret_masked_ids = query_masked_ids = None
            ret_masked_pos = query_masked_pos = None
            ret_masked_weights = query_masked_weights = None
        num_equ_ids = ret_copy_eqids = query_copy_qsids = extend_oov_query_qsids = None

        if self.is_train:
            return [input_ids, segment_ids, input_mask, mask_qkv, query_masked_ids, query_masked_pos, query_masked_weights, ret_masked_ids, 
                    ret_masked_pos, ret_masked_weights, -1, self.task_idx, analogy_attention_mask, position_ids, num_equ_ids, ret_copy_eqids, 
                    query_copy_qsids, extend_oov_query_qsids, oov_tokens]
        else:
            return [input_ids, segment_ids, input_mask, mask_qkv, self.task_idx, analogy_attention_mask, position_ids, num_equ_ids, ret_copy_eqids, 
                    query_copy_qsids, extend_oov_query_qsids, oov_tokens]