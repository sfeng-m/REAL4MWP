
Code for EMNLP 2021 Paper "Recall and Learn: A Memory-augmented Solver for Math Word Problems".

environment: python3.6, pytorch1.2
You can create virtual environment with required package by "conda env create -f REAL.yaml".

preprocess data:
    python3 memory_module.py

train:
    python3 run.py --is_train --train_batch_size 6 --num_train_epochs 80 \
        --start_lr_decay_epoch 40 --dataset math23k --topk 1 \
        --add_copynet --add_memory_module --is_equ_norm 

test:
    python3 run.py --eval_batch_size 6  --dataset math23k --topk 1 \
        --add_copynet --add_memory_module --is_equ_norm 

The project supports multi GPU parallel by setting CUDA_VISIBLE_DEVICES.

top1 result:


topk result:
Although REAL is trained with only a retrieved question, we still have the flexibility to adjust the number of retrieved questions 
at the testing stage by modifying the value of topk, which can affect the model’s performance. 


In addition, It's simple to train topk retrieved questions by modifying the value of topk at training stage, which can obtain better result, though it is not show in our paper.
For example, We train top3 retrieved questions and test top3 retrieved questions, the result is:



Acknowledgments
Our code is based on unilm(补链接). We thank the authors for their wonderful open-source efforts. We use the same license as unilm.