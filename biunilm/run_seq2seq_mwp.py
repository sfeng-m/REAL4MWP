"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import math
import json
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import shutil
import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import sys
o_path = os.getcwd()
sys.path.append('..')

from nn.data_parallel import DataParallelImbalance
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling_mwp import BertForPreTrainingLossMask
import biunilm.seq2seq_loader_mwp as seq2seq_loader

#改变标准输出的默认编码
# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None
    

def main(args, logger):

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    args.log_dir = args.log_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.is_train:
        raise ValueError("`is_train` must be True.")

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)  # BertTokenizer实例化，from_pretrained作用是下载词表；add by sfeng

    vocabs = list(tokenizer.vocab.keys())

    # if args.add_analogy_embedding:
    #     tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
    if args.local_rank == 0:
        dist.barrier()

    if args.is_train:
        if args.add_memory_module:
            bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(args.max_pred, list(tokenizer.vocab.keys(
                )), tokenizer.convert_tokens_to_ids, args.max_seq_length, truncate_config={'max_len_a': args.max_len_a, 'max_len_b': args.max_len_b}, 
                max_analogy_len=args.max_analogy_len, is_train=True)]
        else:
            bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seqBaseline(args.max_pred, list(tokenizer.vocab.keys(
                )), tokenizer.convert_tokens_to_ids, args.max_seq_length, truncate_config={'max_len_a': args.max_len_a, 'max_len_b': args.max_len_b}, 
                max_analogy_len=args.max_analogy_len, is_train=True)]

        train_dataset = seq2seq_loader.Seq2SeqDataset(
            args.memory_train_file, args.train_batch_size, data_tokenizer, is_single_char=args.is_single_char, 
            add_memory_module=args.add_memory_module, bi_uni_pipeline=bi_uni_pipeline, topk=args.topk, is_train=args.is_train)
    
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
            _batch_size = args.train_batch_size
        else:
            train_sampler = DistributedSampler(train_dataset)
            _batch_size = args.train_batch_size // dist.get_world_size()
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                       num_workers=args.num_workers, collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False, drop_last=True) 

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    # t_total = int(math.ceil(len(train_dataset.ex_list) / args.train_batch_size)
    t_total = int(len(train_dataloader) * args.num_train_epochs /
                  args.gradient_accumulation_steps)

    print('train_dataloader:%s' % train_dataloader)
    print('len of train_dataloader:%s' % len(train_dataloader))

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)

    cls_num_labels = 2
    if args.add_memory_module:
        type_vocab_size = 6 + \
            (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 4
    else:
        type_vocab_size = 6 + \
            (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2

    num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
    relax_projection = 4 if args.relax_projection else 0

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if recover_step is None:
        model_recover = {} if args.from_scratch else None
        global_step = 0
    else:
        logger.info("***** Recover model: %d *****", recover_step)
        model_recover = torch.load(os.path.join(
            args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
        global_step = math.floor(
            recover_step * t_total / args.num_train_epochs)

    model = BertForPreTrainingLossMask.from_pretrained(
        args.bert_model, state_dict=model_recover, num_labels=cls_num_labels, num_rel=0,
        type_vocab_size=type_vocab_size, config_path=args.config_path, task_idx=3,
        num_sentlvl_labels=num_sentlvl_labels, max_position_embeddings=args.max_position_embeddings,
        label_smoothing=args.label_smoothing, fp32_embedding=args.fp32_embedding,
        relax_projection=relax_projection, new_pos_ids=args.new_pos_ids, ffn_type=args.ffn_type,
        hidden_dropout_prob=args.hidden_dropout_prob, attention_probs_dropout_prob=args.attention_probs_dropout_prob, 
        num_qkv=args.num_qkv, add_memory_module=args.add_memory_module, max_len_a=args.max_len_a, add_copynet=args.add_copynet,
        vocabs=vocabs, topk=args.topk
        )

    if args.local_rank == 0:
        dist.barrier()

    model.to(device)

    if not args.is_debug:
        if args.local_rank != -1:
            try:
                from torch.nn.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("DistributedDataParallel")
            model = DDP(model, device_ids=[
                        args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        elif n_gpu > 1:
            # model = torch.nn.DataParallel(model)
            model = DataParallelImbalance(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.used_bertAdam:
        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=t_total)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-6) # param_optimizer

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.is_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)

        lr_decay_speed = 2

        model.train()
        if recover_step:
            start_epoch = recover_step+1
        else:
            start_epoch = 1
        for i_epoch in trange(start_epoch, int(args.num_train_epochs)+1, desc="Epoch", disable=args.local_rank not in (-1, 0)):
            # decay half learning rate for 5 epoch. (after 30 epochs.)
            if i_epoch > args.start_lr_decay_epoch and i_epoch % 5 == 0:
                # lr_decay_speed = min(64, lr_decay_speed*2)
                lr_decay_speed = lr_decay_speed*2
                logger.info('learning rate epoch is:{}, speed is:{}'.format(i_epoch, lr_decay_speed))

            if args.local_rank != -1:
                train_sampler.set_epoch(i_epoch)
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)',
                            disable=args.local_rank not in (-1, 0))

            logger.info('iter_bar:%s' % iter_bar)
            logger.info('len of iter_bar:{}'.format(len(iter_bar)))

            all_loss = 0
            
            for step, (batch, oov_tokens) in enumerate(iter_bar):
                batch = [
                    t.to(device) if t is not None else None for t in batch]

                input_ids, segment_ids, input_mask, mask_qkv, query_masked_ids, query_masked_pos, query_masked_weights, \
                    ret_masked_ids, ret_masked_pos, ret_masked_weights, is_next, task_idx, analogy_attention_mask, position_ids, \
                    num_equ_ids, extend_oov_query_qsids, extra_zeros = batch
                if not args.add_num_equ_ids:
                    num_equ_ids = None
                loss = model(input_ids, segment_ids, input_mask, ret_masked_ids, is_next, masked_pos=ret_masked_pos, masked_weights=ret_masked_weights, 
                                task_idx=task_idx, masked_pos_2=query_masked_pos, masked_weights_2=query_masked_weights, masked_labels_2=query_masked_ids, 
                                mask_qkv=mask_qkv, analogy_attention_mask=analogy_attention_mask, step=step, position_ids=position_ids, 
                                num_equ_ids=num_equ_ids, extend_oov_query_qsids=extend_oov_query_qsids, extra_zeros=extra_zeros)

                if n_gpu > 1:    # mean() to average on multi-gpu.
                    loss = loss.mean()
                all_loss += loss.item()
                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss=%5.3f)' % round(all_loss/(step+1),6))

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if i_epoch > args.start_lr_decay_epoch:
                        # modify learning rate with special warm up BERT uses
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.learning_rate / lr_decay_speed  # decay half lr.
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Save a trained model
            param_optimizer = list(model.named_parameters())
            if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                logger.info(
                    "** ** * Saving fine-tuned model and optimizer ** ** * ")
                logger.info("epoch is:{}, loss is:{:.4f}".format(i_epoch, round(all_loss/(step+1),6)))
                model_to_save = model.module if hasattr(
                    model, 'module') else model  # Only save the model it-self
                # 删除之前的模型，防止占用太多内存
                if args.is_delete_early_model:
                    if os.path.exists(args.output_dir):
                        shutil.rmtree(args.output_dir)
                    os.makedirs(args.output_dir)

                if i_epoch == 1 or i_epoch % 5 == 0 or i_epoch >= 70 or args.save_every_epoch:  # 每十个epoch保存一次
                    output_model_file = os.path.join(
                        args.output_dir, "model.{0}.bin".format(i_epoch))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_optim_file = os.path.join(
                        args.output_dir, "optim.{0}.bin".format(i_epoch))
                    torch.save(optimizer.state_dict(), output_optim_file)


                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
