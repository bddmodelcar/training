# Computer Setup

The purpose of this document is to establish a standard set of procedures for setting up the servers in the BDD Model Car project. 

**FOR ADMINS ONLY**

## Network Setup (IF AND ONLY IF, NOT AT DATACENTER COLOCATION)

#### Find the Mac Address and Register it with Berkeley

```
ifconfig -a
```
To get the mac address (of the form XX:XX:XX:XX where XX are in hex). Then register this mac address at:

```
https://netreg.berkeley.edu/
```

## Software and Drivers
#### Install CUDA 8.0
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```
#### Install cuDNN 5.1
- Download CUDNN from https://developer.nvidia.com/rdp/cudnn-download#a-collapseTwo (Make an Account)
```
tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
#### Assorted utilities: 
- Install Cuda Profiling Tools
    ```
    sudo apt-get install libcupti-dev
    ```
#### Install All Dependencies for Running Training Code
- Install docker from https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce (use the amd64 version)
- Install nvidia-enabled docker:
    ```
    wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb 
    sudo dpkg -i nvidia-docker_1.0.1-1_amd64.deb
    ```
- Install Python Docker Environment for our Code:
    ```
    sudo docker pull tpankaj/bdd-pytorch  # will take awhile
    ```

## Misc. Setup
#### Update Ubuntu
```
sudo apt-get update        # Fetches the list of available updates
sudo apt-get upgrade       # Strictly upgrades the current packages
sudo apt-get dist-upgrade  # Installs updates (new ones)
```

#### Setup SSH Server
- Install OpenSSH Server and Curl
    ```
    sudo apt-get install openssh-server curl
    ```
- Get secure sshd_config from @sauhaardac 's GitHub Gists
    ```
    wget https://gist.githubusercontent.com/sauhaardac/017f6b26b3826299109f8be42aa1602e/raw/7b895873720af196535a4f56720aefdd5b7f25e1/sshd_config
    sudo rm /etc/ssh/sshd_config
    sudo mv sshd_config /etc/ssh/sshd_config
    ```
- Start ssh server
    ```
    sudo service ssh restart
    ```

#### Establish IP Address for computer:
- Ask network adminstrator for static IP
- Set IP Address following [this guide](http://www.configserverfirewall.com/ubuntu-linux/ubuntu-set-static-ip-address/)

#### Securing Computer from Attacks
- Install UFW
    ```
    sudo apt-get install ufw
    sudo ufw disable
    sudo ufw default deny
    sudo ufw allow 1022 # for ssh, open ports for other services as well
    sudo ufw allow http
    sudo ufw allow https
    sudo ufw enable
    ```
- Set Strict Password Policy following [this guide](https://help.ubuntu.com/16.04/serverguide/user-management.html)

#### Setup Public Key for Each User (IMPORTANT)
```
git clone https://github.com/bddmodelcar/users_allowed_keys
cd users_allowed_keys
sudo chmod +x install_all_users.sh
sudo ./install_all_users.sh
```
Some warning/error messages may pop up, it's usually just best to ignore them.
