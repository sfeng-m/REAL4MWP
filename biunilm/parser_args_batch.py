
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--bert_model", default='bert-base-chinese', type=str, dest='bert_model',
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_path", default=None, type=str, dest='config_path',
                        help="Bert config file path.")
    parser.add_argument("--output_dir",
                        default='',
                        type=str,
                        dest='output_dir',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default='',
                        type=str,
                        dest='log_dir',
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default='',
                        type=str,
                        dest='model_recover_path',
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--optim_recover_path",
                        default=None,
                        type=str,
                        dest='optim_recover_path',
                        help="The file of pretraining optimizer.")

    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        dest='max_seq_length',
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        dest='do_train',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        dest='do_eval',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        dest='do_lower_case',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=12,
                        type=int,
                        dest='train_batch_size',
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=12,
                        type=int,
                        dest='eval_batch_size',
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, dest='learning_rate',
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, dest='label_smoothing',
                        help="label smoothing for loss.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        # default=0,
                        type=float,
                        dest='weight_decay',
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        dest='finetune_decay',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=80,
                        type=float,
                        dest='num_train_epochs',
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        dest='warmup_proportion',
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, dest='hidden_dropout_prob',
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, dest='attention_probs_dropout_prob',
                        help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        dest='no_cuda',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        dest='local_rank',
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        dest='seed',
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        dest='gradient_accumulation_steps',
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        type=bool,
                        dest='fp16',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true', dest='fp32_embedding',
                        help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0, dest='loss_scale',
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp',
                        default=False,
                        type=bool,
                        dest='amp',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true', dest='from_scratch',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids',
                        default=False,  # 原本为True
                        type=bool,
                        dest='new_segment_ids',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true', dest='new_pos_ids',
                        help="Use new position ids for LMs.")
    parser.add_argument('--tokenized_input', action='store_true', dest='tokenized_input',
                        help="Whether the input is tokenized.")
    parser.add_argument('--max_len_a', type=int, 
                        default=192,
                        dest='max_len_a',
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, 
                        default=64,
                        dest='max_len_b',
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='a', dest='trunc_seg',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', 
                        default=True,
                        type=bool,
                        dest='always_truncate_tail',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float, dest='mask_prob',
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float, dest='mask_prob_eos',
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, 
                        default=64,
                        dest='max_pred',
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int, dest='num_workers',
                        help="Number of workers for the data loader.")  # 使用多进程加载的进程数，0代表不使用多进程； add by sfeng

    parser.add_argument('--mask_source_words', action='store_true', dest='mask_source_words',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0, dest='skipgram_prb',
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1, dest='skipgram_size',
                        help='the max size of ngram mask')
    parser.add_argument('--mask_whole_word',
                        default=True,
                        type=bool,
                        dest='mask_whole_word',
                        help="Whether masking a whole word.")
    parser.add_argument('--do_l2r_training', action='store_true', dest='do_l2r_training',
                        help="Whether to do left to right training")
    parser.add_argument('--has_sentence_oracle', action='store_true', dest='has_sentence_oracle',
                        help="Whether to have sentence level oracle for training. "
                             "Only useful for summary generation")
    parser.add_argument('--max_position_embeddings', type=int, 
                        # default=256,
                        default=512,  # 当analogy添加position embedding的时候，此处必须由256改成512
                        dest='max_position_embeddings',
                        help="max position embeddings")
    parser.add_argument('--relax_projection', action='store_true', dest='relax_projection',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--ffn_type', default=0, type=int, dest='ffn_type',
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int, dest='num_qkv',
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true', dest='seg_emb',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--s2s_special_token', action='store_true', dest='s2s_special_token',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true', dest='s2s_add_segment',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true', dest='s2s_share_segment',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")

    parser.add_argument('--is_train',  action='store_true', dest='is_train',
                        help="train or valid.")
    parser.add_argument('--add_copynet', action='store_true', dest='add_copynet',
                        help="add copynet module")
    parser.add_argument('--add_memory_module', action='store_true', dest='add_memory_module',
                        help="whether add memory module")
    parser.add_argument('--is_equ_norm', action='store_true', dest='is_equ_norm',
                        help="whether use equation normalization.")
    parser.add_argument('--is_debug', action='store_true', dest='is_debug',
                        help="whether debug, use parallel if false")
    parser.add_argument('--used_bertAdam', action='store_true', dest='used_bertAdam',
                        help="optimizer use bertAdam or adam.")
    parser.add_argument('--is_delete_early_model', action='store_true', dest='is_delete_early_model',
                        help="is delete early model")
    parser.add_argument('--easy_to_hard', action='store_true', dest='easy_to_hard',
                        help="where pred question from easy to hard.")

    parser.add_argument('--pos_shift', default=True,
                        type=bool,
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--is_single_char', type=bool, 
                        default=False,
                        dest='is_single_char',
                        help="whether use single char(split diagit)")
    
    parser.add_argument('--topk', type=int, 
                        default=1,
                        dest='topk',
                        help="retrieve top k")
    parser.add_argument('--max_analogy_len', type=int, 
                        default=512,
                        dest='max_analogy_len',
                        help="max analogy token length")
    
    parser.add_argument('--save_every_epoch', type=bool, default=False, dest='save_every_epoch',
                        help="save_every_epoch")
    
    parser.add_argument('--add_num_equ_ids', type=int, default=True, dest='add_num_equ_ids',
                        help="whether to add num_equ_ids.")
    parser.add_argument('--num_equ_size', type=int, default=3, dest='num_equ_size',
                        help="size that number and equation.")
    parser.add_argument('--Fold', type=int, default=1, dest='Fold',
                        help="whether to use 5 fold cross validation and use which fold.")
    parser.add_argument('--start_lr_decay_epoch', type=int, default=40, dest='start_lr_decay_epoch',
                        help="the epoch starting lr decay.")
    parser.add_argument('--dataset', type=str, default='math23k', dest='dataset',
                        help="which dataset, math23k/ape210k.")

    parser.add_argument("--memory_train_file",
                        default='',
                        type=str,
                        dest='memory_train_file',
                        help="memory train file(include question_equation and retrieve question_equation).")
    parser.add_argument("--memory_valid_file",
                        default='',
                        type=str,
                        dest='memory_valid_file',
                        help="memory valid file(include question_equation and retrieve question_equation).")
    
    # decoding parameters
    parser.add_argument('--subset', type=int, default=0, dest='subset',
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--split", type=str, default='test', dest='split',
                        help="Data split (train/val/test).")  
    parser.add_argument('--beam_size', type=int, default=1, dest='beam_size',
                    help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0, dest='length_penalty',
                    help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', default=False, type=bool, dest='forbid_duplicate_ngrams')
    parser.add_argument('--forbid_ignore_word', type=str, default='.', dest='forbid_ignore_word',
    # parser.add_argument('--forbid_ignore_word', type=str, default='', dest='forbid_ignore_word',
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int, dest='min_len')
    parser.add_argument('--need_score_traces', default=True,
                        type=bool, dest='need_score_traces')
    parser.add_argument('--ngram_size', type=int, default=3, dest='ngram_size')
    parser.add_argument('--mode', default="s2s", dest='mode',
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, 
                        default=64, dest='max_tgt_length',
                        help="maximum length of target sequence")
    parser.add_argument('--not_predict_token', type=str, default=None, dest='not_predict_token',
                    help="Do not predict the tokens during decoding.")

    args = parser.parse_args()
    return args