#!/bin/bash

echo "running experiments."

python meta-testing_batch.py  --dataset=ucihar  --steps=50 --runs=5 --plot  --lr=5e-4 --scenario 'nc' --new_seed 

python meta-testing_batch.py  --dataset=hapt  --steps=50 --runs=5 --lr=0.003 --plot --scenario 'nc' --new_seed 

python meta-testing_batch.py  --dataset=dsads  --steps=50 --runs=5 --lr=0.001  --plot --scenario 'nc' --new_seed

python meta-testing_batch.py  --dataset=pamap2  --steps=50 --runs=5 --lr=0.001  --plot --scenario 'nc' --new_seed 



python meta-testing_batch.py  --dataset=ucihar  --steps=50 --runs=5 --plot  --lr=5e-4 --scenario 'nc' --new_seed --augmentation ['Jitter','Scale','Perm','TimeW','MagW']

python meta-testing_batch.py  --dataset=hapt  --steps=50 --runs=5 --lr=0.003 --plot --scenario 'nc' --new_seed --augmentation ['Jitter','Scale','Perm','TimeW','MagW']

python meta-testing_batch.py  --dataset=dsads  --steps=50 --runs=5 --lr=0.001  --plot --scenario 'nc' --new_seed --augmentation ['Jitter','Scale','Perm','TimeW','MagW']

python meta-testing_batch.py  --dataset=pamap2  --steps=50 --runs=5 --lr=0.001  --plot --scenario 'nc' --new_seed  --augmentation ['Jitter','Scale','Perm','TimeW','MagW']

