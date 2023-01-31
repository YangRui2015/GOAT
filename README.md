# OOD Generalization for Offline Goal-conditioned RL (GOAT)
Code for GOAT, a new weighted imitation-based method to improve OOD generalization for offline GCRL.

We provide offline goal-conditioned benchmark with offline dataset in the 'offline_data' folder, including 'random' and 'expert' settings. The 'buffer.pkl' is used for WGCSL and other algorithms included in our codes (GCSL, MARVIL, BC, HER, DDPG, Actionable Models), and each item in the buffer are also provided as *.npy files for training Goal BCQ and Goal CQL. Due to the storage limitation, the full offline dataset is in this anonymous google drive link: https://drive.google.com/drive/folders/1SIo3qFmMndz2DAnUpnCozP8CpG420ANb.


<!-- 
<div style="text-align: center;">
<img src="pic/offline_hard_tasks.png" >
</div> -->


## Requirements
python3.6+, tensorflow, gym, mujoco, mpi4py

## Installation
- Clone the repo and cd into it

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage
Environments: PointFixedEnv-v1, FetchReach, FetchSlide, FetchPick, HandReach.

GOAT:  
```bash
python -m goat.run --env FetchReach   --mode goat   --su_method exp_adv_2_clip10_baw_tanstd_norm01  --offline_train --load_path ./offline_data/FetchReach/   --load_buffer --log_path ${path_name}    --save_path ${path_name}
```
For GOAT+expectile regression, please use mode='goat_ER'


WGCSL: 
```bash
python -m  goat.run  --env=FetchReach  --mode supervised --su_method exp_adv_2_clip10_baw  --load_path ./offline_data/FetchReach/   --offline_train  --load_buffer  --log_path ${path_name}
```


GCSL:
```bash
python -m  goat.run  --env=FetchReach  --mode supervised --load_path ./offline_data/FetchReach/   --offline_train  --load_buffer
```


MARVIL+HER
```bash
python -m  goat.run  --env=FetchReach  --mode supervised  --load_path ./offline_data/FetchReach/ --load_buffer --offline_train  --su_method exp_adv_2_clip10
```

BC
```bash
python -m  goat.run  --env=FetchReach  --mode supervised  --load_path ./offline_data/FetchReach/ --load_buffer --offline_train   --no_relabel
```

DDPG+HER
```bash
python -m  goat.run  --env=FetchReach  --mode her  --load_path ./offline_data/FetchReach/ --load_buffer --offline_train   
```

CQL+HER
```bash
python -m  goat.run  --env=FetchReach  --mode conservation  --load_path ./offline_data/FetchReach/ --load_buffer --offline_train 
```

### Ablations
Replcing mode and su_method with the following arguments.
GOAT+ER:  
```bash
--mode goat_ER   --su_method exp_adv_2_clip10_baw_tanstd_norm01
```

GOAT:  
```bash
--mode goat   --su_method exp_adv_2_clip10_baw_tanstd_norm01
```

WGCSL+Ensemble
```bash
--mode ensemble_supervised   --su_method exp_adv_2_clip10_baw
```

