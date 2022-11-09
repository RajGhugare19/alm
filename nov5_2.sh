python train.py id=HalfCheetah-v2 seed=2 num_train_steps=300000 positive_reward=True total_positive_reward=True wandb_log=True wandb_run_name=pos_rew_2& 

python train.py id=HalfCheetah-v2 seed=3 num_train_steps=300000 positive_reward=True total_positive_reward=True wandb_log=True wandb_run_name=pos_rew_3& 

python train.py id=HalfCheetah-v2 num_train_steps=300000 linear_classifier=True seed=1 wandb_log=True wandb_run_name=alm_lin_class_1& 

python train.py id=HalfCheetah-v2 num_train_steps=300000 linear_classifier=True seed=2 wandb_log=True wandb_run_name=alm_lin_class_2& 

python train.py id=HalfCheetah-v2 num_train_steps=300000 linear_classifier=True seed=3 wandb_log=True wandb_run_name=alm_lin_class_3&