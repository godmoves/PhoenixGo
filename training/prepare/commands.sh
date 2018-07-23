# update your system
sudo apt-get update
sudo apt-get upgrade

# common tools
sudo apt-get install git -y
sudo apt-get install vim -y
sudo apt-get install tmux -y

# get and install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
sudo sh Anaconda3-5.2.0-Linux-x86_64.sh

# updata conda
conda update conda

# get cuda
wget https://developer.download.nvidia.com/compute/cuda/9.0/secure/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb

# install cuda
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda -y

# add cuda environment variables
echo "
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda" >> $HOME/.bashrc
source $HOME/.bashrc

# get cudnn
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-linux-x64-v7.tgz

# install cudnn
tar -zxvf cudnn-9.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/ -d
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# get tensorrt
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/3.0/ga/nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb

# install tensorrt
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
sudo apt-get update
sudo apt-get install tensorrt -y
sudo apt-get install python3-libnvinfer-doc -y
sudo apt-get install uff-converter-tf -y

# verify the installation
dpkg -l | grep TensorRT

# install tensorflow-gpu
pip install tensorflow-gpu
