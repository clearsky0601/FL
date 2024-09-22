# DFRD

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.12.1

- scipy==1.10.1

- numpy==1.23.5

- sklearn==1.2.2

- pandas==2.0.0

- mpi4py==3.1.1

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- LightFed # experiments for baselines, DFRD and datasets
    |-- experiments/ #
        |-- datasets/ 
            |-- data_distributer.py/  # the load datasets,including FMNSIT, SVHN, CIFAR-10, CIFAR-100, Tiny-ImageNet, FOOD101
        |-- horizontal/ ## DFRD and baselines
            |-- DENSE/
            |-- DFRD/
            |-- fedavg/
            |-- fedFTG/
            |-- PT_method/
        |-- models
            |-- model.py/  ##load backnone architectures
    |-- lightfed/  
        |-- core # important configure
        |-- tools
```

## Run pipeline for Run pipeline for DFRD
1. Entering the DFRD
```python
cd LightFed
cd experiments
cd horizontal
cd DFRD
```

2. You can run any models implemented in `main_DFRD.py`. For examples, you can run our model on `SVHN` dataset by the script:
```python
python main_DFRD.py --batch_size 64 --I 20 --comm_round 200 --model_heterogeneity True /
                    --model_split_mode random --model_here_level 5 --mask False --mask_ensemble_gen True --mask_ensemble_glo True / 
                    --lr_lm 0.01 --global_lr 0.01 --gen_lr 0.0002 --b1 0.5 --b2 0.999 / 
                    --adv_I 10 --gen_I 2 --global_I 5 --latent_dim 64 --beta_tran 1.0 --beta_div 1.0 --beta_ey 0.25 --lambda_ 0.5 --alpha 0.5 --gamma 1.0 / 
                    --noise_label_combine mul --data_partition_mode non_iid_dirichlet_unbalanced --non_iid_alpha 0.1 --client_num 10 --selected_client_num 10 --seed 0 /
                    --model_type Lenet --data_set SVHN --eval_batch_size 256 --device cuda
```
And you can run other baselines, such as 
```python
cd LightFed
cd experiments
cd horizontal
cd fedavg
python main_fedavg.py --batch_size 64 --I 20 --comm_round 100 --lr_lm 0.008 --mask False /
                      --data_partition_mode non_iid_dirichlet_unbalanced --non_iid_alpha 0.01 /
                      --client_num 10 --selected_client_num 10 --seed 0 /
                      --model_type Lenet --data_set SVHN --eval_batch_size 256 --device cuda
```

