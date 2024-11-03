# Meta-training for Human-Activity Recogniton (HAR)

These codes implements OML (Online aware Meta-Learning) and MAML-Rep approaches to HAR datasests (PAMAP2, HAPT, UCIHAR, and DSADS)

This approach is based on the paper:

Meta-Learning Representations for Continual Learning by Khurram Javed and Martha White

Paper : https://arxiv.org/abs/1905.12588

The original OML code can be found at  
https://github.com/khurramjaved96/mrcl

The code provided by the authors were used as a base to support:

1) new scenarios of continual learning: nc (new classes) 
2) HAR/time series datasets
3) pipeline to experiment execution, configurable to different combination of parameteres
4) set of plots, including stats per class, iteration, average, etc.

# Main files:
 
|File                         |Description                               |
|-------------------------------|----------------------------------------|
|meta-training.py            | RLN training                              |
|meta-testing.py             | meta-testing                              |
|run_meta-testing.py         | runs meta-testing                         |
|run_encoders_nc.sh          | runs encoders training                    |
|run_meta-testing_batch.sh   | runs meta-testing in batch                |


# Main directories
 
|Folder                         |Description                          |
|-------------------------------|-------------------------------------|
|configs          |parametrization files                              |
|datasets         | classes and codes for dataset and benchmark preprocessing  |
|model| classes for model setting and meta-learning                   |
|oml|oml model setting                                                |
|utils   |main experiments' common classes with various purposes      |

# Execution examples

**meta-training - encoder training - RLN**

python  meta-training.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset     --new_seed  --runs=5 --random  --model 'oml'

**meta-testing**

python meta-testing.py  --model 'maml' --path </home/>  --plot --plot_file plot_meta-testing.py --runs 20 --classes_schedule 2 --new_seed --reset_weights --iid

**batch **

python meta-testing_batch.py --dataset=pamap2 --steps=50 --runs=5 --lr=0.001 --plot --scenario nc --new_seed --augmentation [Jitter,Scale,Perm,TimeW,MagW]


**notes**
 - Parametrization description can be found  in ../configs/classification/class_parser* files
 - Examples of sh file to run meta-training in run_encoders_nc.sh  
 - Examples of sh file to run batch baseline in run_meta-testing_batch.sh





