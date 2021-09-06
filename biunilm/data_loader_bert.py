"""
在原版seq2seq_loader.py的基础上对mask方式进行修改，具体是：将词组作为一个整体，同时进行mask，让模型更好的学习语义信息；
"""
from random import randint, shuffle, choice
from random import random as rand
import math
import torch
import jieba

from biunilm.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]  # 第一个元素记录句子头部被删除的词个数，第二个元素记录具体尾部被删除的词个数，下同； add by sfeng
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()     # 前面为赋值语句，删除trunc_tokens中的元素即删除tokens_a/tokens_b的元素，pop默认删除最后一个元素； add by sfeng
            num_truncated[1] += 1  # num_truncated_a/num_truncated_b尾部被删除个数加1； add by sfeng
    return num_truncated_a, num_truncated_b


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[], multi_task=False):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        self.ex_list = []
        if file_oracle is None:
            for i in range(1):  # 通过复制10份数据，将静态mask改为动态mask；
                with open(file_src, "r", encoding='utf-8') as f_src, open(file_tgt, "r", encoding='utf-8') as f_tgt:
                    for src, tgt in zip(f_src, f_tgt):
                        src = src.replace('[SPLIT]', '叕')
                        if multi_task:
                            src, sample_label, _ = src.split('#')
                            sample_label = int(float(sample_label))
                        else:
                            sample_label = None
                        src_tk = tokenizer.tokenize(src.strip())
                        tgt_tk = tokenizer.tokenize(tgt.strip())
                        assert len(src_tk) > 0
                        assert len(tgt_tk) > 0
                        self.ex_list.append((src_tk, tgt_tk, sample_label))
                f_src.close()
                f_tgt.close()
                print('data copy:', i, len(self.ex_list))
        else:
            with open(file_src, "r", encoding='utf-8') as f_src, \
                    open(file_tgt, "r", encoding='utf-8') as f_tgt, \
                    open(file_oracle, "r", encoding='utf-8') as f_orc:
                for src, tgt, orc in zip(f_src, f_tgt, f_orc):
                    src_tk = tokenizer.tokenize(src.strip())
                    tgt_tk = tokenizer.tokenize(tgt.strip())
                    s_st, labl = orc.split('\t')
                    s_st = [int(x) for x in s_st.split()]
                    labl = [int(x) for x in labl.split()]
                    self.ex_list.append((src_tk, tgt_tk, s_st, labl))
        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

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

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, 
                mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s", has_oracle=False, 
                num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False, src_split=False, 
                perf_label_dict=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.src_split = src_split  # add by sfeng
        self.max_len_stu_str_src = 10
        self.max_len_stu_src = 80
        self.max_len_stu_tgt = 150
        self.perf_label_dict = perf_label_dict

    def __call__(self, instance):
        tokens_a, tokens_b, sample_label = instance[:3]  # tokens_a, tokens_b分别表示token化的原句子和目标句子（一句话）; sample_label表示原句子是否通顺的标签； add by sfeng

        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b

        # -3  for special tokens [CLS], [SEP], [SEP]
        # （控制tokens_a的长度，self.max_len-3即腾出三个位置给[CLS], [SEP], [SEP]这三个字符，分别安插在句首，一二句中间，句尾）add by sfeng
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # if self.src_split:
        #     # 对每个点评项添加分隔符(源数据中已有分隔符，这里把它改成[S2S_SEP])； add by sfeng;
        #     tokens_a = ['[S2S_SEP]' if to_a == '叕' else to_a for to_a in tokens_a]

        # 该方法在数据存在英文的情况下行不通
        # split_tokens_a = ''.join(tokens_a).split('叕')
        # perf_label, perf_label_mask = [], []
        # perf_label_map = {'学业水平': 0, '思想品德': 1, '综合素养': 2, '其他': 3}
        # stu_com_tokens = []
        # stu_com_ids = []
        # for tok_a in split_tokens_a:
        #     stu_com = [ta for ta in tok_a]
        #     if tok_a in self.perf_label_dict:
        #         perf_label_mask.append(1)
        #         p_label = self.perf_label_dict[tok_a]
        #         perf_label.append(perf_label_map[p_label])
        #     else:
        #         perf_label_mask.append(0)
        #         perf_label.append(3)
        #     if len(stu_com) >= self.max_len_stu_str_src:
        #         stu_com = stu_com[:self.max_len_stu_str_src]
        #     else:
        #         n_pad = self.max_len_stu_str_src - len(stu_com)
        #         stu_com.extend(['[PAD]']*n_pad)
        #     stu_com_tokens.extend(stu_com)
        #     stu_com_id = self.indexer(stu_com)
        #     stu_com_ids.extend(stu_com_id)

        tokens_a = ['[COM_SEP]' if to_a == '叕' else to_a for to_a in tokens_a]  # [COM_SEP]表示点评项分隔符
        stu_com_tokens = []
        stu_com_ids = []
        stu_com = []
        perf_label, perf_label_mask = [], []
        perf_label_map = {'学业水平': 0, '思想品德': 1, '综合素养': 2, '其他': 3}
        for i, tok_a in enumerate(tokens_a):
            if tok_a == '[COM_SEP]' or i == len(tokens_a)-1:
                if tok_a == tokens_a[-1]:
                    stu_com.append(tok_a)
                if ''.join(stu_com) in self.perf_label_dict:
                    perf_label_mask.append(1)
                    p_label = self.perf_label_dict[''.join(stu_com)]
                    perf_label.append(perf_label_map[p_label])
                else:
                    perf_label_mask.append(0)
                    perf_label.append(3)
                if len(stu_com) >= self.max_len_stu_str_src-1:
                    stu_com = ['[COM_CLS]'] + stu_com[:self.max_len_stu_str_src-1]
                else:
                    n_pad = self.max_len_stu_str_src - len(stu_com) - 1
                    stu_com.insert(0, '[COM_CLS]')
                    stu_com.extend(['[PAD]']*n_pad)
                stu_com_tokens.extend(stu_com)
                stu_com_id = self.indexer(stu_com)
                stu_com_ids.extend(stu_com_id)
                stu_com = []
            else:
                stu_com.append(tok_a)

        stu_com_ids = stu_com_ids[:self.max_len_stu_src]
        classifier_attention_mask = [1] * len(stu_com_ids)
        classifier_attention_mask.extend([0]*(self.max_len_stu_src - len(stu_com_ids)))
        stu_com_ids.extend([0]*(self.max_len_stu_src - len(stu_com_ids)))
        tokens_a = stu_com_tokens[:self.max_len_stu_src]
        # tokens_a.extend(['[PAD]']*(self.max_len_stu_src - len(tokens_a)))
        tokens_b = tokens_b[:self.max_len_stu_tgt-3]

        max_len_perf_label = int(self.max_len_stu_src/self.max_len_stu_str_src)
        perf_label_mask = perf_label_mask[:max_len_perf_label]
        perf_label_mask.extend([0]*(max_len_perf_label - len(perf_label_mask)))
        perf_label = perf_label[:max_len_perf_label]
        perf_label.extend([3]*(max_len_perf_label - len(perf_label)))

        com_cls_idx = []
        for i, tok_a in enumerate(tokens_a):
            if tok_a == '[COM_CLS]':
                com_cls_idx.append(i)
        com_cls_idx = com_cls_idx[:max_len_perf_label]
        com_cls_idx.extend([0]*(max_len_perf_label - len(com_cls_idx)))

        # Add Special Tokens
        if self.s2s_special_token:
            tokens = ['[S2S_CLS]'] + tokens_a + \
                ['[S2S_SEP]'] + tokens_b + ['[SEP]']
        else:
            tokens = ['[CLS]'] + tokens_a + ['[S2S_SEP]'] + tokens_b + ['[SEP]']

        # tokens_no_masked = ['[CLS]'] + tokens_b + ['[SEP]']
        # input_ids_no_masked = self.indexer(tokens_no_masked)
        input_ids_no_masked = self.indexer(tokens)
        s2s_sep_idx = [len(tokens_a) + 1]

        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        if self.src_split:
            # 对每个点评项添加标志位（不同点评项的segment_id不一样）； add by sfeng;
            src_segment_ids = []
            si = 0
            for tok in tokens:
                src_segment_ids.append(si)
                if tok in ['[S2S_SEP]', '[SEP]']:
                    si += 1
            # src_segment_ids += [si+1]*(len(tokens_b)+1)

        if self.pos_shift:
            n_pred = min(self.max_pred, len(tokens_b))
            masked_pos = [len(tokens_a)+2+i for i in range(len(tokens_b))]
            masked_weights = [1]*n_pred
            masked_ids = self.indexer(tokens_b[1:]+['[SEP]'])
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)
            if self.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.max_pred, max(
                1, int(round(effective_length*self.mask_prob))))
            # candidate positions of masked tokens

            # 将词组作为一个整体，同时进行mask
            unk_idx = [[idx, tok] for idx, tok in enumerate(tokens_b) if len(tok) > 1]
            line = [tok for idx, tok in enumerate(tokens_b) if len(tok) == 1]
            line = ''.join(line)

            line = jieba.lcut(line)
            tokens_b_tmp = []
            for li in line:
                for i, w in enumerate(li):
                    if i == 0:
                        tokens_b_tmp.append(w)
                    else:
                        tokens_b_tmp.append('##' + w)
            for idx, tok in unk_idx:
                tokens_b_tmp.insert(idx, tok)

            assert len(tokens_b) == len(tokens_b_tmp), "len(tokens_b) must equal len(tokens_b_tmp)..."
            tokens_phrase = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b_tmp + ['[SEP]']

            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens_phrase):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('##')):  # 最后一个条件过滤掉词组除了第一字之后的其他字，目的是为了tokens_b中的字与词组被选中的概率相等；
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                    cand_pos.append(i)
                elif i < len(tokens_a)+2:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens_phrase[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens_phrase)) and tokens_phrase[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]  # target 部分被mask掉的词的真实标签； add by sfeng
            # print('tokens_phrase:', tokens_b_tmp)
            # print('masked_tokens:', masked_tokens)
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens)

            # Token Indexing
            masked_ids = self.indexer(masked_tokens)  # target中需要mask部分的token对应的id； add by sfeng
        # Token Indexing
        input_ids = self.indexer(tokens)

        # TODO：加上token_b的内容
        comment_tokens = tokens[len(tokens_a)+2:]
        # comment_tokens = ['[CLS]'] + ['[SEP]'] + comment_tokens + ['[SEP]']
        comment_ids = [input_ids[0]] + input_ids[len(tokens_a)+1:]
        if len(comment_ids) > 150:
            # comment_ids = comment_ids[:149] + [comment_ids[-1]]
            comment_ids = comment_ids[:150]
        tgt_n_pad = self.max_len_stu_tgt - len(comment_ids)
        comment_ids.extend([0]*tgt_n_pad)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        input_ids_no_masked.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        if self.src_split:
            src_segment_ids.extend([0]*n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask_no_masked = torch.ones(self.max_len, self.max_len, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            second_st, second_end = len(
                tokens_a)+2, len(tokens_a)+len(tokens_b)+3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            st, end = 0, len(tokens_a) + len(tokens_b) + 3
            input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        oracle_pos = None
        oracle_weights = None
        oracle_labels = None
        if self.has_oracle:
            s_st, labls = instance[2:]
            oracle_pos = []
            oracle_labels = []
            for st, lb in zip(s_st, labls):
                st = st - num_truncated_a[0]
                if st > 0 and st < len(tokens_a):
                    oracle_pos.append(st)
                    oracle_labels.append(lb)
            oracle_pos = oracle_pos[:20]
            oracle_labels = oracle_labels[:20]
            oracle_weights = [1] * len(oracle_pos)
            if len(oracle_pos) < 20:
                x_pad = 20 - len(oracle_pos)
                oracle_pos.extend([0] * x_pad)
                oracle_labels.extend([0] * x_pad)
                oracle_weights.extend([0] * x_pad)

            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx,
                    oracle_pos, oracle_weights, oracle_labels)
        if self.src_split:
            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, sample_label, self.task_idx,
            src_segment_ids)
        else:
            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, sample_label, self.task_idx,
                    input_ids_no_masked, input_mask_no_masked, stu_com_ids, comment_ids, perf_label, perf_label_mask, 
                    s2s_sep_idx, classifier_attention_mask, com_cls_idx)  # _no_masked表示不进行masked操作；


class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s", num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False, tokenizer=None):
        super().__init__()
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.max_len_stu_str_src = 10
        self.max_len_stu_src = 80
        self.tokenizer = tokenizer

    def __call__(self, instance):
        tokens_a, max_a_len = instance
        if '#' in tokens_a:
            tokens_a = tokens_a[:tokens_a.index('#')]  # tokens_a 已被分割成一个个字符；

        max_a_len = self.max_len_stu_src
        tokens_a = ['[COM_SEP]' if to_a == '叕' else to_a for to_a in tokens_a]  # [COM_SEP]表示点评项分隔符
        # print('tokens_a:', ''.join(tokens_a))
        stu_com_tokens = []
        stu_com_ids = []
        stu_com = []
        for i, tok_a in enumerate(tokens_a):
            if tok_a == '[COM_SEP]' or i == len(tokens_a)-1:
                if tok_a == tokens_a[-1]:
                    stu_com.append(tok_a)
                if len(stu_com) >= self.max_len_stu_str_src-1:
                    stu_com = ['[COM_CLS]'] + stu_com[:self.max_len_stu_str_src-1]
                else:
                    n_pad = self.max_len_stu_str_src - len(stu_com) - 1
                    stu_com.insert(0, '[COM_CLS]')
                    stu_com.extend(['[PAD]']*n_pad)
                stu_com_tokens.extend(stu_com)
                stu_com_id = self.indexer(stu_com)
                stu_com_ids.extend(stu_com_id)
                stu_com = []
            else:
                stu_com.append(tok_a)
        # print('stu_com_ids:', stu_com_ids)
        stu_com_ids = stu_com_ids[:self.max_len_stu_src]
        classifier_attention_mask = [1] * len(stu_com_ids)
        classifier_attention_mask.extend([0]*(self.max_len_stu_src - len(stu_com_ids)))
        stu_com_ids.extend([0]*(self.max_len_stu_src - len(stu_com_ids)))
        tokens_a = stu_com_tokens[:self.max_len_stu_src]
        s2s_sep_idx = [len(tokens_a) + 1]
        # tokens_a.extend(['[PAD]']*(self.max_len_stu_src - len(tokens_a)))
        # print('tokens_a after add pad:', ''.join(tokens_a))

        max_len_perf_label = int(self.max_len_stu_src/self.max_len_stu_str_src)
        com_cls_idx = []
        for i, tok_a in enumerate(tokens_a):
            if tok_a == '[COM_CLS]':
                com_cls_idx.append(i)
        com_cls_idx = com_cls_idx[:max_len_perf_label]
        com_cls_idx.extend([0]*(max_len_perf_label - len(com_cls_idx)))

        # Add Special Tokens
        if self.s2s_special_token:
            padded_tokens_a = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]']
        else:
            padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            if self.mode == "s2s":
                _enc_seg1 = 0 if self.s2s_share_segment else 4
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [
                            0] + [1]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                    else:
                        segment_ids = [
                            4] + [6]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                else:
                    segment_ids = [4]*(len(padded_tokens_a)) + \
                        [5]*(max_len_in_batch - len(padded_tokens_a))
            else:
                segment_ids = [2]*max_len_in_batch
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(padded_tokens_a)) + [1] * \
                (max_len_in_batch - len(padded_tokens_a))
        else:
            mask_qkv = None

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)
        # print('tokens:', ''.join(tokens))
        # print('token_id2token:', ''.join(self.tokenizer.convert_ids_to_tokens(input_ids)).split('[COM_SEP]'))
        # print('...'*30)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        return (input_ids, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx, stu_com_ids, s2s_sep_idx,
                classifier_attention_mask, com_cls_idx)


class Preprocess4Classify(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False, src_split=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.src_split = src_split  # add by sfeng

    def __call__(self, instance):
        tokens_a, tokens_b, sample_label = instance[:3]  # tokens_a, tokens_b分别表示token化的原句子和目标句子（一句话）; sample_label表示原句子是否通顺的标签； add by sfeng

        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b

        # -3  for special tokens [CLS], [SEP], [SEP]
        # （控制tokens_a的长度，self.max_len-3即腾出三个位置给[CLS], [SEP], [SEP]这三个字符，分别安插在句首，一二句中间，句尾）add by sfeng
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        if self.src_split:
            # 对每个点评项添加分隔符(源数据中已有分隔符，这里把它改成[S2S_SEP])； add by sfeng;
            tokens_a = ['[S2S_SEP]' if to_a == '叕' else to_a for to_a in tokens_a]

        # tokens_a = ['[S2S_SEP]' if to_a == '叕' else to_a for to_a in tokens_a]

        # Add Special Tokens
        if self.s2s_special_token:
            tokens = ['[S2S_CLS]'] + tokens_a + \
                ['[S2S_SEP]'] + tokens_b + ['[SEP]']
        else:
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        if self.src_split:
            # 对每个点评项添加标志位（不同点评项的segment_id不一样）； add by sfeng;
            src_segment_ids = []
            si = 0
            for tok in tokens:
                src_segment_ids.append(si)
                if tok in ['[S2S_SEP]', '[SEP]']:
                    si += 1
            # src_segment_ids += [si+1]*(len(tokens_b)+1)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        if self.src_split:
            src_segment_ids.extend([0]*n_pad)

        input_mask = torch.ones(self.max_len, self.max_len, dtype=torch.long)

        if self.src_split:
            return (input_ids, segment_ids, input_mask, sample_label, src_segment_ids)
        else:
            return (input_ids, segment_ids, input_mask, sample_label)

