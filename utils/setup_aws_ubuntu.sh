#!/bin/bash
echo "Adding conda to bashrc"
echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc

echo "Changing CUDA Version to 10.0"
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda

echo "Installing Gluon environment"
conda env create -f ./gluon_cu100.yml
