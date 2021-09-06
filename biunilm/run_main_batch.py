from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
o_path = os.getcwd()
sys.path.append('..')

from utils.Logger import initlog
from .parser_args_batch import get_args
from .run_seq2seq_mwp import main as train_main
from .decoder_seq2seq_mwp import main_generation

# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
Epoch = 65

def main():
    args = get_args()
    if args.dataset == 'math23k':
        OUTPUT_DIR = './comment/math23k/output/fine_tune/'
        args.log_dir = OUTPUT_DIR + 'bert_log'
        # for fold in [1,2,3,4,5]:
        for fold in [1]:
            args.Fold = fold
            if args.is_equ_norm:
                args.memory_train_file = './preprocess/sim_result/sim_question_by_w2v_train_math23k_equNorm_{}fold_top5.pkl'.format(args.Fold)
                args.memory_valid_file = './preprocess/sim_result/sim_question_by_w2v_valid_math23k_equNorm_{}fold_top5.pkl'.format(args.Fold)
                if args.add_copynet and args.add_memory_module:
                    model_name = 'REAL_math23k_{}fold_CP2_ME_EN_top1'.format(args.Fold)
                elif not args.add_copynet and args.add_memory_module: 
                    model_name = 'REAL_math23k_{}fold_woCP_ME_EN_top1'.format(args.Fold)
                elif not args.add_copynet and not args.add_memory_module: 
                    model_name = 'REAL_math23k_{}fold_woCP_woME_EN_top1'.format(args.Fold)
            else:
                args.memory_train_file = './preprocess/sim_result/sim_question_by_w2v_train_math23k_{}fold_top5.pkl'.format(args.Fold)
                args.memory_valid_file = './preprocess/sim_result/sim_question_by_w2v_valid_math23k_{}fold_top5.pkl'.format(args.Fold)
                if args.add_copynet and args.add_memory_module:
                    model_name = 'REAL_math23k_{}fold_CP2_ME_woEN_top1'.format(args.Fold)
                if not args.add_copynet and not args.add_memory_module: 
                    model_name = 'REAL_math23k_{}fold_woCP_woME_woEN_top1'.format(args.Fold)

            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)
            log_path = model_name + '.log'
            args.output_dir = OUTPUT_DIR + model_name
            args.model_recover_path = args.output_dir + '/model.epoch.bin'

            logger = initlog(logfile=args.log_dir + "/" + log_path)
            logger.info('pid:{}, epoch:{}, args:{}'.format(os.getpid(), Epoch, args))
            if args.is_train:
                train_main(args, logger)
            else:
                main_generation(args, logger, i_epoch=Epoch)

    elif args.dataset == 'ape210k':
        OUTPUT_DIR = './comment/ape210k/output/fine_tune/'
        args.log_dir = OUTPUT_DIR + 'bert_log'
        if args.is_equ_norm:
            args.memory_train_file = './preprocess/sim_result/sim_question_by_w2v_train_ape210k_equNorm_top5.pkl'
            args.memory_valid_file = './preprocess/sim_result/sim_question_by_w2v_test_ape210k_equNorm_top5.pkl'
            if args.add_copynet and args.add_memory_module:
                model_name = 'REAL_ape210k_CP2_ME_EN_top1'
            elif not args.add_copynet and args.add_memory_module: 
                model_name = 'REAL_ape210k_woCP_ME_EN_top1'
            elif not args.add_copynet and not args.add_memory_module: 
                model_name = 'REAL_ape210k_woCP_woME_EN_top1'
        else:
            args.memory_train_file = './preprocess/sim_result/sim_question_by_w2v_train_ape210k_top5.pkl'
            args.memory_valid_file = './preprocess/sim_result/sim_question_by_w2v_test_ape210k_top5.pkl'
            if args.add_copynet and args.add_memory_module:
                model_name = 'REAL_ape210k_CP2_ME_woEN_top1'
            if not args.add_copynet and not args.add_memory_module: 
                model_name = 'REAL_ape210k_woCP_woME_woEN_top1'

        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        log_path = model_name + '.log'
        args.output_dir = OUTPUT_DIR + model_name
        args.model_recover_path = args.output_dir + '/model.epoch.bin'
        
        logger = initlog(logfile=args.log_dir + "/" + log_path)
        logger.info('pid:{}, epoch:{}, args:{}'.format(os.getpid(), Epoch, args))
        if args.is_train:
            train_main(args, logger)
        else:
            main_generation(args, logger, i_epoch=Epoch)


if __name__ == "__main__":
    main()

    

    