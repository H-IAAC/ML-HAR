#!/bin/bash

echo "running experiments."



# pamap2 nc = 70k nic = 25k
# with and without augmentation



#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --random  --model 'oml'
#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=10000  --plot --reset  --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=10000  --plot --reset  --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --rerset    --new_seed  --runs=5 --random  --model 'oml' --main_folder 'new_benchmark_timestep_pooled'

python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset    --new_seed  --runs=5 --random  --model 'oml' --main_folder 'new_benchmark_final_timestep_v2' --standardization_mode 'timestep'


#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset  --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=pamap2 --scenario=nc --steps=25000  --plot --reset  --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'


# ucihar nc = 150k nic = 30k
# with and without augmentation

#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset    --new_seed  --runs=5 --random --model 'oml' --main_folder 'new_benchmark_final'

#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=10000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']   --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=10000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']   --new_seed  --runs=5 --model 'maml'


#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset    --new_seed  --runs=5 --random --model 'oml' --main_folder 'new_benchmark'

#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']   --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=ucihar --scenario=nc --steps=30000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']   --new_seed  --runs=5 --model 'maml'


# hapt nc = 25k nic = 25k
# with and without augmentation


#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset    --new_seed  --runs=5 --random --model 'oml' --main_folder 'new_benchmark_final'

#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=10000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=10000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'


#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset      --new_seed  --runs=5 --random --model 'oml' --main_folder 'new_benchmark'
#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=hapt --scenario=nc --steps=25000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'



# dsads nc =10k nic = 10k
# with and without augmentation


#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=10000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=10000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'



#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --random --model 'oml' --main_folder 'new_benchmark_timestep_pooled'

#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=10000  --plot --reset    --new_seed  --runs=5 --random --model 'oml' --main_folder 'new_benchmark_final'

#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=20000  --plot --reset    --new_seed  --runs=5 --model 'maml'

#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=20000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --random --model 'oml'
#python offline_OML_benchmark.py  --dataset=dsads --scenario=nc --steps=20000  --plot --reset   --augmentation ['Jitter','Scale','Perm','TimeW','MagW']  --new_seed  --runs=5 --model 'maml'


: '
exemplo comentário
'
