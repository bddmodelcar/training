# Computer Monitoring

## Run once per client machine

### Install distributed shell
#### Linux
```bash
sudo apt-get --assume-yes install dsh
```
#### Mac OSX
http://macappstore.org/dsh/


### Setup Config Files
#### Now create a `~/.ssh/config` file and add all of the machines to it. Here is an example:
```
Host bdd3.tspankaj.com 
    User sauhaarda
    Port 1022
Host bdd6.tspankaj.com 
    User sauhaarda
    Port 1022
Host bdd5.neuro.berkeley.edu
    User sauhaarda
    Port 1022
Host bdd4.dyn.berkeley.edu
    User sauhaarda
    Port 1022
```

#### Create a `~/.dsh/machines.list` file. In this file put in all of the host names from your ssh config. Here is an example:
```
bdd3.tspankaj.com
bdd6.tspankaj.com
bdd4.dyn.berkeley.edu
bdd5.neuro.berkeley.edu
```

#### Finally add an alias to allow for even easier access for monitoring computers:
```bash
echo "alias monitorall='dsh -Mac -- remote-log'" >> ~/.bashrc

## Finished!
```
Now you'll be able to run `monitorall` anytime you want to see the status of each computer. The output will look something like this:
```
bdd5.dyn.berkeley.edu:   PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
bdd5.dyn.berkeley.edu: 30193 root      20   0 30.638g 6.504g 528376 R  93.3  5.2  23:02.27 python
bdd5.dyn.berkeley.edu: Wed Aug 16 21:10:31 2017       
bdd5.dyn.berkeley.edu: +-----------------------------------------------------------------------------+
bdd5.dyn.berkeley.edu: | NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
bdd5.dyn.berkeley.edu: |-------------------------------+----------------------+----------------------+
bdd5.dyn.berkeley.edu: | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
bdd5.dyn.berkeley.edu: | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
bdd5.dyn.berkeley.edu: |===============================+======================+======================|
bdd5.dyn.berkeley.edu: |   0  GeForce GTX 108...  Off  | 0000:4B:00.0     Off |                  N/A |
bdd5.dyn.berkeley.edu: | 28%   38C    P8    16W / 250W |      2MiB / 11172MiB |      0%      Default |
bdd5.dyn.berkeley.edu: +-------------------------------+----------------------+----------------------+
bdd5.dyn.berkeley.edu: |   1  GeForce GTX 108...  Off  | 0000:4C:00.0     Off |                  N/A |
bdd5.dyn.berkeley.edu: | 55%   79C    P2    81W / 250W |   1877MiB / 11170MiB |     22%      Default |
bdd5.dyn.berkeley.edu: +-------------------------------+----------------------+----------------------+
bdd5.dyn.berkeley.edu:                                                                                
bdd5.dyn.berkeley.edu: +-----------------------------------------------------------------------------+
bdd5.dyn.berkeley.edu: | Processes:                                                       GPU Memory |
bdd5.dyn.berkeley.edu: |  GPU       PID  Type  Process name                               Usage      |
bdd5.dyn.berkeley.edu: |=============================================================================|
bdd5.dyn.berkeley.edu: |    1     30193    C   python                                        1875MiB |
bdd5.dyn.berkeley.edu: +-----------------------------------------------------------------------------+
bdd3.tspankaj.com:   PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
bdd3.tspankaj.com: 10854 bala      20   0 40.226g 0.012t 166016 R  86.7  9.9 449:50.57 python
bdd3.tspankaj.com: 10875 bala      20   0 41.614g 0.014t 165668 R 100.0 11.0 523:46.08 python
bdd3.tspankaj.com: 11149 bala      20   0 29.662g 6.465g  86512 R  80.0  5.1 499:47.42 python
bdd3.tspankaj.com: 11393 bala      20   0 30.072g 6.391g  86568 R  73.3  5.1 463:29.35 python
bdd3.tspankaj.com: 31902 bala      20   0 29.506g 6.038g  87740 R  86.7  4.8 171:49.60 python
bdd3.tspankaj.com: 10854 bala      20   0 40.226g 0.012t 166016 R  66.7  9.9 449:51.16 python
bdd3.tspankaj.com: 10875 bala      20   0 41.614g 0.014t 165668 R  81.2 11.0 523:46.79 python
bdd3.tspankaj.com: Wed Aug 16 21:10:32 2017       
bdd3.tspankaj.com: +-----------------------------------------------------------------------------+
bdd3.tspankaj.com: | NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
bdd3.tspankaj.com: |-------------------------------+----------------------+----------------------+
bdd3.tspankaj.com: | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
bdd3.tspankaj.com: | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
bdd3.tspankaj.com: |===============================+======================+======================|
bdd3.tspankaj.com: |   0  TITAN X (Pascal)    Off  | 0000:4C:00.0     Off |                  N/A |
bdd3.tspankaj.com: | 41%   69C    P2   184W / 250W |   6103MiB / 12189MiB |     74%      Default |
bdd3.tspankaj.com: +-------------------------------+----------------------+----------------------+
bdd3.tspankaj.com: |   1  TITAN X (Pascal)    Off  | 0000:4D:00.0     Off |                  N/A |
bdd3.tspankaj.com: | 51%   84C    P2   104W / 250W |   3728MiB / 12186MiB |     17%      Default |
bdd3.tspankaj.com: +-------------------------------+----------------------+----------------------+
bdd3.tspankaj.com:                                                                                
bdd3.tspankaj.com: +-----------------------------------------------------------------------------+
bdd3.tspankaj.com: | Processes:                                                       GPU Memory |
bdd3.tspankaj.com: |  GPU       PID  Type  Process name                               Usage      |
bdd3.tspankaj.com: |=============================================================================|
bdd3.tspankaj.com: |    0     10854    C   python                                         315MiB |
bdd3.tspankaj.com: |    0     10875    C   python                                         315MiB |
bdd3.tspankaj.com: |    0     11149    C   python                                        1565MiB |
bdd3.tspankaj.com: |    0     11393    C   python                                        2067MiB |
bdd3.tspankaj.com: |    0     31902    C   python                                        1839MiB |
bdd3.tspankaj.com: |    1     10854    C   python                                        1863MiB |
bdd3.tspankaj.com: |    1     10875    C   python                                        1863MiB |
bdd3.tspankaj.com: +-----------------------------------------------------------------------------+
bdd6.tspankaj.com:   PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
bdd6.tspankaj.com: 10323 tpankaj   20   0 45.916g 0.021t  89460 R 553.3 34.8   7462:51 python
bdd6.tspankaj.com: 10412 tpankaj   20   0 45.972g 0.022t  89444 R 546.7 36.0   7469:08 python
bdd6.tspankaj.com: 23431 tpankaj   20   0 33.389g 0.011t  88680 R  46.7 17.7 280:30.31 python
bdd6.tspankaj.com: 17937 root      20   0  218564  11612  10968 S   0.0  0.0   0:12.94 Xorg
bdd6.tspankaj.com: Wed Aug 16 21:10:32 2017       
bdd6.tspankaj.com: +-----------------------------------------------------------------------------+
bdd6.tspankaj.com: | NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
bdd6.tspankaj.com: |-------------------------------+----------------------+----------------------+
bdd6.tspankaj.com: | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
bdd6.tspankaj.com: | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
bdd6.tspankaj.com: |===============================+======================+======================|
bdd6.tspankaj.com: |   0  GeForce GTX 108...  Off  | 0000:4B:00.0     Off |                  N/A |
bdd6.tspankaj.com: | 47%   67C    P2    82W / 250W |   4034MiB / 11172MiB |      9%      Default |
bdd6.tspankaj.com: +-------------------------------+----------------------+----------------------+
bdd6.tspankaj.com: |   1  GeForce GTX 108...  Off  | 0000:4C:00.0      On |                  N/A |
bdd6.tspankaj.com: | 40%   58C    P8    30W / 250W |     55MiB / 11169MiB |      0%      Default |
bdd6.tspankaj.com: +-------------------------------+----------------------+----------------------+
bdd6.tspankaj.com:                                                                                
bdd6.tspankaj.com: +-----------------------------------------------------------------------------+
bdd6.tspankaj.com: | Processes:                                                       GPU Memory |
bdd6.tspankaj.com: |  GPU       PID  Type  Process name                               Usage      |
bdd6.tspankaj.com: |=============================================================================|
bdd6.tspankaj.com: |    0     10323    C   python                                        1585MiB |
bdd6.tspankaj.com: |    0     10412    C   python                                        1585MiB |
bdd6.tspankaj.com: |    0     23431    C   python                                         861MiB |
bdd6.tspankaj.com: |    1     17937    G   /usr/lib/xorg/Xorg                              50MiB |
bdd6.tspankaj.com: +-----------------------------------------------------------------------------+
bdd4.dyn.berkeley.edu:   PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
bdd4.dyn.berkeley.edu:  1334 root      20   0  219436  51404  36972 S   0.0  0.0   0:01.68 Xorg
bdd4.dyn.berkeley.edu: Wed Aug 16 21:10:32 2017       
bdd4.dyn.berkeley.edu: +-----------------------------------------------------------------------------+
bdd4.dyn.berkeley.edu: | NVIDIA-SMI 375.82                 Driver Version: 375.82                    |
bdd4.dyn.berkeley.edu: |-------------------------------+----------------------+----------------------+
bdd4.dyn.berkeley.edu: | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
bdd4.dyn.berkeley.edu: | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
bdd4.dyn.berkeley.edu: |===============================+======================+======================|
bdd4.dyn.berkeley.edu: |   0  TITAN X (Pascal)    Off  | 0000:4B:00.0     Off |                  N/A |
bdd4.dyn.berkeley.edu: | 23%   29C    P8     8W / 250W |      1MiB / 12189MiB |      0%      Default |
bdd4.dyn.berkeley.edu: +-------------------------------+----------------------+----------------------+
bdd4.dyn.berkeley.edu: |   1  TITAN X (Pascal)    Off  | 0000:4C:00.0      On |                  N/A |
bdd4.dyn.berkeley.edu: | 23%   33C    P8    10W / 250W |     43MiB / 12186MiB |      0%      Default |
bdd4.dyn.berkeley.edu: +-------------------------------+----------------------+----------------------+
bdd4.dyn.berkeley.edu:                                                                                
bdd4.dyn.berkeley.edu: +-----------------------------------------------------------------------------+
bdd4.dyn.berkeley.edu: | Processes:                                                       GPU Memory |
bdd4.dyn.berkeley.edu: |  GPU       PID  Type  Process name                               Usage      |
bdd4.dyn.berkeley.edu: |=============================================================================|
bdd4.dyn.berkeley.edu: |    1      1334    G   /usr/lib/xorg/Xorg                              40MiB |
bdd4.dyn.berkeley.edu: +-----------------------------------------------------------------------------+
```
