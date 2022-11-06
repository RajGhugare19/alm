python train.py id=HalfCheetah-v2 lambda_cost=0.05 seed=1 num_train_steps=300000 wandb_log=True wandb_run_name=alm_lambda0.05_1& 

python train.py id=HalfCheetah-v2 lambda_cost=0.05 seed=2 num_train_steps=300000 wandb_log=True wandb_run_name=alm_lambda0.05_2& 

python train.py id=HalfCheetah-v2 lambda_cost=0.05 seed=3 num_train_steps=300000 wandb_log=True wandb_run_name=alm_lambda0.05_3&

python train.py id=Ant-v2 lambda_cost=0.05 seed=1 wandb_log=True wandb_run_name=alm_lambda0.05_1& 

python train.py id=Ant-v2 lambda_cost=0.05 seed=2 wandb_log=True wandb_run_name=alm_lambda0.05_2& 

python train.py id=Ant-v2 lambda_cost=0.05 seed=3 wandb_log=True wandb_run_name=alm_lambda0.05_3