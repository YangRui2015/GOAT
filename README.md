# Generalizable Offline goAl-condiTioned RL (GOAT)

Code for the ICML 2023 paper "What is Essential for Unseen Goal Generalization of Offline Goal-conditioned RL?". GOAT is a new weighted supervised method to improve OOD generalization for offline GCRL. This repo is based on [baselines](https://github.com/openai/baselines) and [WGCSL](https://github.com/YangRui2015/AWGCSL).


We provide the benchmark for OOD generaliztion of offline GCRL with offline datasets in the 'offline_data' folder. Due to the storage limitation, we only include PointReach, FetchReach, FetchPush, FetchPick datasets in this repo, and the full offline dataset can be found in this google drive [link](https://drive.google.com/drive/folders/1Q8bWoRVuNZrMsjvymIiagIXwvmQeVqMO?usp=share_link).




## Requirements
python3.6+, tensorflow, gym, mujoco, mpi4py

## Installation
- Clone the repo and cd into it

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage
Environments: PointFixedEnv-v1 (for the didactic example), FetchReach, FetchPush, FetchPick, FetchSlide, HandReach.

Corresponding OOD testing environments are automatically selected for evaluation after training.

### GOAT

```bash
CUDA_VISIBLE_DEVICES=${gpu} python -m goat.run --env FetchReach   --mode goat   --su_method exp_adv_2_clip10_baw_tanstd_norm01  --offline_train --load_path ./offline_data/FetchReach/{pkl_name}   --load_buffer --log_path ${path_name}    --save_path ${path_name}
```
For GOAT+expectile regression, please use mode='goat_ER'

### Baselines

1. WGCSL: 
```bash
python -m  goat.run  --env=FetchReach  --mode supervised --su_method exp_adv_2_clip10_baw  --load_path ./offline_data/FetchReach/{pkl_name}   --offline_train  --load_buffer  --log_path ${path_name}
```


2. GCSL:
```bash
python -m  goat.run  --env=FetchReach  --mode supervised --load_path ./offline_data/FetchReach/{pkl_name}   --offline_train  --load_buffer
```



3. BC
```bash
python -m  goat.run  --env=FetchReach  --mode supervised  --load_path ./offline_data/FetchReach/{pkl_name} --load_buffer --offline_train   --no_relabel
```

4. MARVIL+HER
```bash
python -m  goat.run  --env=FetchReach  --mode supervised  --load_path ./offline_data/FetchReach/{pkl_name} --load_buffer --offline_train  --su_method exp_adv_2_clip10
```

5. DDPG+HER
```bash
python -m  goat.run  --env=FetchReach  --mode her  --load_path ./offline_data/FetchReach/{pkl_name} --load_buffer --offline_train   
```

6. CQL+HER
```bash
python -m  goat.run  --env=FetchReach  --mode conservation  --load_path ./offline_data/FetchReach/{pkl_name} --load_buffer --offline_train 
```

7. MSG+HER
```bash
python -m  goat.run  --env=FetchReach  --mode MSG  --load_path ./offline_data/FetchReach/{pkl_name} --load_buffer --offline_train 
```

### Ablations
Replcing '--mode' and '--su_method' with the following arguments.

1. GOAT+Expectile Regression:  
```bash
--mode goat_ER   --su_method exp_adv_2_clip10_baw_tanstd_norm01
```

2. GOAT:  
```bash
--mode goat   --su_method exp_adv_2_clip10_baw_tanstd_norm01
```

3. WGCSL+Ensemble
```bash
--mode ensemble_supervised   --su_method exp_adv_2_clip10_baw
```

4. WGCSL, WGCSL w/o DSW (i.e., MARVIL+HER), GCSL, BC are presented in the above section. 

5. $\chi^2$-divergence: replace the 'exp_adv' in the '--su_method' flag with 'adv'.

6. V versions: use '--mode goat_V' for V version of GOAT and '--mode supervised_V' for V version of WGCSL.


### Online Fine-tuning
First, you can save the model trained by different algorithms. Second, you can load the model and fine-tune the model with the following arguments. We do not load the offline dataset for fine-tuning in this work.

Online fine-tuning with DDGP+HER
```bash
python -m goat.run --env ${env_name}  --num_env 1 --random_init 0 --mode her  --load_path ${load_path} --load_model --log_path ${log_path}    --save_path ${log_path}
```
Please make sure that the pre-trained agents are not trained by V versions when fine-tuning using DDPG+HER.


## Citation
If you find GOAT helpful for your work, please cite:
```
@article{yang2023essential,
  title={What is Essential for Unseen Goal Generalization of Offline Goal-conditioned RL?},
  author={Yang, Rui and Lin, Yong and Ma, Xiaoteng and Hu, Hao and Zhang, Chongjie and Zhang, Tong},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
