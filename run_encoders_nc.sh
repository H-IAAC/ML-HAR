#!/bin/bash

echo "running experiments."



# pamap2 
# with and without augmentation


python  meta-training.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset     --new_seed  --runs=5 --random  --model 'oml'
python  meta-training.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset     --new_seed  --runs=5 --model 'maml'

python  meta-training.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
python  meta-training.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'


# ucihar 
# with and without augmentation


python  meta-training.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset     --new_seed  --runs=5 --random --model 'oml'
python  meta-training.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset     --new_seed  --runs=5 --model 'maml'

python  meta-training.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset    --augmentation ['Jitter','Scale','Perm','TimeW','MagW']   --new_seed  --runs=5 --random --model 'oml'
python  meta-training.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset    --augmentation ['Jitter','Scale','Perm','TimeW','MagW']   --new_seed  --runs=5 --model 'maml'


# hapt 
# with and without augmentation



python  meta-training.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset       --new_seed  --runs=5 --random --model 'oml'
python  meta-training.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset     --new_seed  --runs=5 --model 'maml'

python  meta-training.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset    --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
python  meta-training.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset    --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'



# dsads 
# with and without augmentation


python  meta-training.py  --dataset=dsads --scenario=nc --steps=20000  --plot --reset     --new_seed  --runs=5 --random --model 'oml'
python  meta-training.py  --dataset=dsads --scenario=nc --steps=20000  --plot --reset     --new_seed  --runs=5 --model 'maml'

python  meta-training.py  --dataset=dsads --scenario=nc --steps=20000  --plot --reset    --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
python  meta-training.py  --dataset=dsads --scenario=nc --steps=20000  --plot --reset    --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'


