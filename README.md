# SRCNN
* modified from [nnUyi/SRGAN](https://github.com/nnUyi/SRGAN)

# Requirements

  - tensorflow 1.3.0
  - python 2.7.12 or python 3.*
  - numpy 1.13.1
  - scipy 0.17.0

# Usages
  ### train data
  - In this repo, I use parts of [ImageNet](http://www.image-net.org/) datasets as **train data**, [here](https://pan.baidu.com/s/1eSJC0lc) you can download the datasets that I used. 
  
  - After you have download the datasets, copy ImageNet(here I only use 3137 images) datsets to ***/data/train***, then you have ***/data/train/ImageNet*** path, and training images are stored in ***/data/train/ImageNet***
  
  - I crop image into **256*256 resolution**, actually you can crop them according to your own.

  ### val data
  - **Set5** dataset is used as **val data**, you can download it [here](https://pan.baidu.com/s/1dFyFFSt).
  
  - After you download **Set5**, please store it in ***/data/val/*** , then you have ***/data/val/Set5*** path, and val images are stored in ***/data/val/Set5***

  ### test data
  - **Set14** dataset is used as **test data**, you can download it [here](https://pan.baidu.com/s/1nvmUkBn).
  
  - After you download **Set14**, please store it in ***/data/test/*** , then you have ***/data/test/Set14*** path, and val images are stored in ***/data/test/Set14***

  ## training

      $ python main.py --is_training=True --is_testing=False

  ## testing

      $ python main.py --is_training=False --is_testing=True --file_name=file_name.png --load_model=SRCNN.model-1


