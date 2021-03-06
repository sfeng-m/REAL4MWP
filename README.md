
Code for EMNLP 2021 Paper "[Recall and Learn: A Memory-augmented Solver for Math Word Problems](https://aclanthology.org/2021.findings-emnlp.68.pdf)".

# environment  
python3.6, pytorch1.2\
You can install related packages directly through "```pip install requirements.txt```" or \
create virtual environment with required package by "```conda env create -f REAL.yaml```".

# preprocess data
    python3 memory_module.py

# train: 
     python3 run.py --is_train --train_batch_size 6 --num_train_epochs 80 \
        --start_lr_decay_epoch 40 --dataset math23k --topk 1 \
        --add_copynet --add_memory_module --is_equ_norm 

# test:
    python3 run.py --eval_batch_size 6  --dataset math23k --topk 1 \
        --add_copynet --add_memory_module --is_equ_norm 

# parallel
The project supports multi GPU parallel by setting CUDA_VISIBLE_DEVICES.

# result
We don't spend more energy on adjusting the parameters of the model. You can get better results through parameter adjustment.
## top1 result:
 
<img width="500" height="300" src="https://github.com/sfeng-m/REAL4MWP/blob/master/images/top1_result.png" />

## topk result:
Although REAL is trained with only a retrieved question, we still have the flexibility to adjust the number of retrieved questions 
at the testing stage by modifying the value of topk, which can affect the model’s performance. 

<img width="500" height="300" src="https://github.com/sfeng-m/REAL4MWP/blob/master/images/topk_result.png" />

In addition, It's simple to train topk retrieved questions by modifying the value of topk at training stage, which can obtain better result, though it is not show in our paper.

# Improve
Further, we improve the effectiveness of REAL model to solve math work problems(MWP) by optimizing the memory module. \
More details please see our NIPS2021 Paper on MATHAI4ED Workshop: ["REAL2: An End-to-end Memory-augmented Solver for Math Word Problems"](https://mathai4ed.github.io/papers/papers/paper_7.pdf). 


# Citation
If the paper or the code helps you, please cite the paper in the following format :
```
@inproceedings{huang-etal-2021-recall-learn,
    title = "Recall and Learn: A Memory-augmented Solver for Math Word Problems",
    author = "Huang, Shifeng and Wang, Jiawei and Xu, Jiao and Cao, Da and Yang, Ming",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021"
    }
```


# Acknowledgments
Our code is based on [unilm](https://github.com/microsoft/unilm/tree/master/unilm-v1) . We thank the authors for their wonderful open-source efforts. We use the same license as unilm.
