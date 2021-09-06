# coding=utf-8

"""
@author: stefan
@Describe: ACC评价指标
"""

import numpy as np
import pandas as pd
import os, re, io, sys

sys.path.append('..')
from process_studata import PREFIX, is_equal

def merge_decimal(pred):
    r_list = []
    i = 0
    while i < len(pred):
        if pred[i] == '.' and len(r_list) > 0:
            r_list[-1] = str(r_list[-1]) + str(pred[i]) + str(pred[i+1])
            i += 2
        else:
            r_list.append(pred[i])
            i += 1
    return r_list


def eval_main(args, logger, preds, tgts, pri_questions):
    Prefix = PREFIX()
    question_total, acc_right, uneval = 0, 0, 0
    for pred, tgt, ques in zip(preds, tgts, pri_questions):
        question_total += 1
        pred = pred.replace('\n', '')
        tgt = tgt.replace('\n', '')
        try:
            pred_equation = eval(pred)
            pred_equation = merge_decimal(pred_equation)
            gen_ans = Prefix.compute_prefix_expression(pred_equation)
            if args.dataset == 'math23k' and args.Fold > 0:
                tgt_equation = merge_decimal(eval(tgt))
                answer = Prefix.compute_prefix_expression(tgt_equation)
            else:
                answer = eval(tgt.replace('[','(').replace(']',')').replace('^','**'))
            if is_equal(gen_ans, answer):
                acc_right += int(is_equal(gen_ans, answer))
        except:
            uneval += 1
    logger.info('acc_right: {}, question_total: {}, uneval: {}, correct score: {:.4f}'.format(acc_right, question_total, uneval, acc_right / question_total))


if __name__ == '__main__':
    eval_main()
