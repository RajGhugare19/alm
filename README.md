# Aligned Latent Models (ALMs)
[Raj Ghugare](https://rajghugare19.github.io/), [Homanga Bharadhwaj](https://homangab.github.io/), [Benjamin Eysenbach](https://ben-eysenbach.github.io/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/). 

<p align="center">
  <img width="32%" src="https://media.giphy.com/media/2TQ2SDfPeJTteqdd8Q/giphy.gif">
  <img width="32%" src="https://media.giphy.com/media/FbdAXTitljv1awVc0o/giphy.gif">
  <img width="32%" src="https://media.giphy.com/media/Pv84wCSt7QOOpU5cRo/giphy.gif">
</p>

## Installation

Install MuJoCo version mjpro150 binaries from their [website](https://www.roboti.us/download.html). Extract the downloaded `mjpro150` directory into `~/.mujoco/`. Download the free activation key from [here](https://www.roboti.us/license.html) and place it in `~/.mujoco/`. Add the following lines in `~/.bashrc` and then source it.<br>
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
```

If you are using the latest versions of MuJoCo (> 2.0), it is possible that it might produce inaccurate or zero contact forces in the Humanoid-v2 and Ant-v2 environments. See [#2593](https://github.com/openai/gym/issues/2593), [#1541](https://github.com/openai/gym/issues/1541) and [ #1636](https://github.com/openai/gym/issues/1636). If you encounter any errors, check the troubleshooting section of [mujoco-py](https://github.com/openai/mujoco-py).

Create virtual environment named `env_alm` using command:<br>
```sh
python3 -m venv env_alm
```

Install all the packages used to run the code using the `requirements.txt` file: <br>
```sh
pip install -r requirements.txt
```

These instructions are for code that was tested to run on Ubuntu 22.04 with Python 3.10.4.

## Training

To train an ALM agent on Humanoid-v2 environment:<br> 
```sh
python train.py id=Humanoid-v2
```

Log training and evaluation details using wandb:<br>
```sh
python train.py id=Humanoid-v2 wandb_log=True
```
To perform the bias evaluation experiments from our paper:<br>
```sh
python train.py id=Humanoid-v2 eval_bias=True
```

## Acknowledgment
Our codebase has been build using/on top of the following codes. We thank the respective authors for their awesome contributions.
- [DRQ-v2](https://github.com/facebookresearch/drqv2)<br>
- [SAC-SVG](https://github.com/facebookresearch/svg)<br>
- [RED-Q](https://github.com/watchernyu/REDQ)

