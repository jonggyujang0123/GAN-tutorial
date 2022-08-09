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
pip install tqdm easydict wandb imageio
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

### License:
This project is licensed under MIT License - see the LICENSE file for details
