# deep-reinforcement-learning-project1
## 1. The Environment
The target of the project is to train an agent to navigate around a square world and collect yellow bananas as many as possible while avoiding the blue ones.

![image](https://github.com/gcbbobo/deep-reinforcement-learning-project1/blob/master/banana.gif)

### Reward:
Collecting a yellow banana: reward +1

Collecting a blue banana:   reward -1

### State:
The state space has 37 dimensions including the velocity of the agent and the position information of surrounding objects.

### Action:
4 discrete actions available:

0 - move forward.

1 - move backward.

2 - turn left.

3 - turn right.

### Target:
An average score of +13 over 100 consecutive episodes.

## 2. Installation
### Anaconda Installation and Virtual Environment Setup:
Follow the instructions listed: 

https://inmachineswetrust.com/posts/deep-learning-setup/

Be sure to work with python 3.x. As far as I can remember, I particularly downloaded python 3.6.8 to work with ML-Agents and I am not sure which versions would also work well.

### CUDA and cuDNN
If GPU available, go to the official website for CUDA and cuDNN.Download packages that corresponding to the GPU. 

For GPU Training using The ML-Agents Toolkit, only CUDA v9.0 and cuDNN v7.0.5 is supported at present.

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md

### TensorFlow: 
For ml agent:

`pip install tensorflow==1.7.1`

`pip install tensorflow-gpu==1.7.1`

### PyTorch:
Follow the official guide:

https://pytorch.org/get-started/locally/

Download the CPU version.

Some commands taken by myself:

`conda install pytorch-cpu torchvision-cpu -c pytorch`

`pip install --user https://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-win_amd64.whl`

`pip install --user torchvision`

My PyTorch version for CPU is 1.0.1.

### Box2d:
`conda install swig` # needed to build Box2D in the pip install

`pip install box2d-py` # a repackaged version of pybox2d

### pip3:
Use conda and cd to the dictionary of any (virtual) environment.

In my case, 2 virtual environments are set up in addition to the root environment. Their locations are:

C:\download\anaconda\envs\dl

C:\download\anaconda\envs\tensorflow2

C:\download\anaconda

Take the first environment as an example.

Open Anaconda Prompt and enter:

`cd C:\download\anaconda\envs\dl`

Then type:

`python -m pip install --upgrade pip --force-reinstall`

Repeat this process for each environment.

### Unity:
https://unity.com/

The version of Unity should be compatible with Python or something interesting may show up like:

mlagents.envs.exception.UnityEnvironmentException: 

The API number is not compatible between Unity and python. Python API : API-7, Unity API : API-4

Someone report similar problems on github for other reasons:

https://github.com/Unity-Technologies/ml-agents/issues/1770

One feasible solution seems to be using "pip3" command instead of "pip" during installation of packages.

Personally, I downloaded Unity 2018.2.8f1 and it works well with all the other packages. 

### ML-Agent:
Thank to Rahul's kind assistance, I just follow some instructions provided by:

https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows

Download all the rest packages with "pip3" command.

The ML-Agent package is also uploaded on my github for reference.



