#!/bin/bash

sudo apt update

#opencv
sudo apt install libopencv-dev -y

#cuda toolkit 
#sudo apt install nvidia-cuda-toolkit -y
#currently this method does not work because cuda 11.5 clashes with gcc 11.3 and compilation fails

#cuda toolkit 11.7
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-11-7 -y
sudo apt install cuda-11-7
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"

#modify makefile to point to the correct library folders and use the appropriate gpu architecture

