# -*- encoding: UTF-8 -*-
"""
@author: 'stefan'
@describe: 论文中的memory module模块，调用检索模块进行question检索；
"""

import pickle
from process_studata import fetch_math23k, fetch_ape210k
from config import arg_config
from preprocess import train_w2v

MODE = 'train'
TopK = 5
FiveFold = True
Used_Transfer_Num = False
Used_Equ_Norm = True

Dataset = 'math23k'  # math23k/math23k_test1000/ape210k


def sim_model(train_data, valid_data, train_save_path, valid_save_path, fold):
    """相似度模型的训练和预测"""
    if MODE == 'train':
        train_w2v.main(data=train_data, mode=MODE, topk=TopK, dataset=Dataset, fold=fold)
    train_question_sim_qsas, valid_question_sim_qsas = train_w2v.main(
            data=[train_data, valid_data], mode='valid', topk=TopK, dataset=Dataset, fold=fold)
    pickle.dump(train_question_sim_qsas, open(train_save_path, 'wb'))
    pickle.dump(valid_question_sim_qsas, open(valid_save_path, 'wb'))


def process_fivefold_data():
    """处理math23k五折交叉验证数据"""

    Fetch = fetch_math23k()
    all_data, equ_ids_dict = Fetch.load_math23k_line_data(filename=arg_config['path_math23k_dataset'], mode='train', single_char=False, used_equ_norm=Used_Equ_Norm)
    
    one_fold_length = int(len(all_data) / 5)
    for i in range(5):
        fold_valid_data = all_data[one_fold_length*i:one_fold_length*(i+1)]
        fold_train_data = all_data[:one_fold_length*i] + all_data[one_fold_length*(i+1):]

        if Used_Equ_Norm:
            train_save_path = arg_config['path_w2v_sim_question_train'].replace('.pkl', '_{}_equNorm_{}fold_top{}.pkl'.format(Dataset, i+1, TopK))
            valid_save_path = arg_config['path_w2v_sim_question_valid'].replace('.pkl', '_{}_equNorm_{}fold_top{}.pkl'.format(Dataset, i+1, TopK))
        else:
            train_save_path = arg_config['path_w2v_sim_question_train'].replace('.pkl', '_{}_{}fold_top{}.pkl'.format(Dataset, i+1, TopK))
            valid_save_path = arg_config['path_w2v_sim_question_valid'].replace('.pkl', '_{}_{}fold_top{}.pkl'.format(Dataset, i+1, TopK))

        sim_model(fold_train_data, fold_valid_data, train_save_path, valid_save_path, fold=i+1)


def process_unfold_data():
    """处理无需进行交叉验证的数据集，包括math23k官方测试集和ape数据集都可用该函数处理"""

    if Dataset == 'Math23k_Test1000':
        Fetch = fetch_math23k()
        train_data, equ_ids_dict = Fetch.load_math23k_line_data(filename=arg_config['path_math23k_all_train'], mode='train', single_char=False, used_equ_norm=Used_Equ_Norm)
        valid_data, _ = Fetch.load_math23k_line_data(filename=arg_config['path_math23k_test_1000'], mode='test', single_char=False, used_equ_norm=Used_Equ_Norm)
    else:
        Fetch = fetch_ape210k()
        train_data, equ_ids_dict = Fetch.load_raw_data(filename=arg_config['path_ape210k_train'], mode='train', single_char=False, used_equ_norm=Used_Equ_Norm)
        valid_data, _ = Fetch.load_raw_data(filename=arg_config['path_ape210k_valid'], mode='test', single_char=False, used_equ_norm=Used_Equ_Norm)
        test_data, _ = Fetch.load_raw_data(filename=arg_config['path_ape210k_test'], mode='test', single_char=False, used_equ_norm=Used_Equ_Norm)

    if Used_Equ_Norm:
        train_save_path = arg_config['path_w2v_sim_question_train'].replace('.pkl', '_{}_equNorm_top{}.pkl'.format(Dataset, TopK))
        valid_save_path = arg_config['path_w2v_sim_question_valid'].replace('.pkl', '_{}_equNorm_top{}.pkl'.format(Dataset, TopK))
        if Dataset == 'ape210k':
            test_save_path = arg_config['path_w2v_sim_question_test'].replace('.pkl', '_{}_equNorm_top{}.pkl'.format(Dataset, TopK))
    else:
        train_save_path = arg_config['path_w2v_sim_question_train'].replace('.pkl', '_{}_top{}.pkl'.format(Dataset, TopK))
        valid_save_path = arg_config['path_w2v_sim_question_valid'].replace('.pkl', '_{}_top{}.pkl'.format(Dataset, TopK))
        if Dataset == 'ape210k':
            test_save_path = arg_config['path_w2v_sim_question_test'].replace('.pkl', '_{}_top{}.pkl'.format(Dataset, TopK))
            
    if Dataset == 'Math23k_Test1000':
        sim_model(train_data, valid_data, train_save_path, valid_save_path, fold=0)
    else:
        sim_model(train_data, test_data, train_save_path, test_save_path, fold=0)


if __name__ == '__main__':
    
    if FiveFold:
        process_fivefold_data()
    else:
        process_unfold_data()
    
