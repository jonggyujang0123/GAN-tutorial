# Generator-Tutorials


## Features
- (multi-node/gpu) (training/validation/test)
- Weight and bias (Wandb) MLops
- Easy Code Structure
- CelebA dataset
- Customizable configuration

## Requirements and Setup

### In your computer
- Install Anaconda

```
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash ~/Desktop/<anacondafile> -b -p 
source anaconda3/bin/activate
```

- Create Environment 
```
conda create -n torch python=3.9
conda activate torch
conda install pytorch==1.12.0 torchvision==0.13.0 -c pytorch
pip install tqdm easydict wandb imageio moviepy
cd canonicalSGD
python setup.py develop
```

### Wandb

- WANDB [Enter this link](https://wandb.ai/site)
1. create your account (you can use your Github account)
2. in `config/*****.yaml` edit wandb setting
3. run our script and type certification code.
4. Done

## Wandb implementation 

<img width="1098" alt="image" src="https://user-images.githubusercontent.com/88477912/183643650-ab42b226-8351-4ccf-ac44-880a88e5e13d.png">

** You can turn off the wandb log by editing the below line**
```
os.environ['WANDB_SILENT'] = "true"
```

# Details:

## Single-GPU 

### Training
```
python tools/main.py --config config/...yaml --resume (0 or 1)
```

### Test

```
python tools/main.py --config config/...yaml --test 1
```

## Multi-GPU 

### Training

```
sh tools/dist_train.sh <#GPUS> <1 for resume>
```

### Test

```
sh tools/dist_test.sh <#GPUS>
```


# GAN

## Results 
![image](https://user-images.githubusercontent.com/88477912/183643008-eaa66d9d-0b56-4736-9112-e9bb9e6ef045.png)
![image](https://user-images.githubusercontent.com/88477912/183643062-1ae8882d-beca-48d4-8440-8dfabd33a24a.png)
![image](https://user-images.githubusercontent.com/88477912/183643110-52e5c405-fb6b-4d53-9d42-c0c6c4bdd359.png)

## Loss functions

<img width="355" alt="image" src="https://user-images.githubusercontent.com/88477912/183643233-fcdc5355-75c2-4836-834a-44fcaff1448a.png">

<img width="347" alt="image" src="https://user-images.githubusercontent.com/88477912/183643254-88271475-9521-4d29-bbbc-f7de1baf8ee2.png">

# VAE


## Results 

### Real vs Fake
![image](https://user-images.githubusercontent.com/88477912/183818262-fb923bca-790d-4d1c-a784-7aaeee312ea2.png)

### Generated data from random latent vectors

![image](https://user-images.githubusercontent.com/88477912/183818322-7501136f-a611-4ae4-8e3b-4b9e3905cab7.png)

# Diffusion Autoencoder

## Generated image from random noise

![gif_28_56c545f9c56657d533d6](https://user-images.githubusercontent.com/88477912/184265033-902068f9-f418-467c-92b1-55ef2cfaa73e.gif)

![image](https://user-images.githubusercontent.com/88477912/184265091-f81e7dd8-cab7-49bd-9e3b-67eec0e866ea.png) 

### License:
This project is licensed under MIT License - see the LICENSE file for details
