import gym
from gym import spaces

import numpy as np

from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms



class mnistEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, digit=2):
        super().__init__()    
        
        self.action_space = spaces.Box(low=-1.,high=1., shape=(1,), dtype=np.float32)   
        self.observation_space = spaces.Box(low=0., high=1., shape=
                        (1, 28, 28), dtype=np.float32)
        
        self.mnist = MNIST(root="/home/apoorva/datasets/MNIST", train=False, download=True)

        self.valid_idx = np.flatnonzero(self.mnist.targets == digit)
        
        self.state = 0. # angle of image
        self.digit = None
        
        self.process_noise = 1.

    def reset(self):
        # Reset the state of the environment to an initial state
        idx = np.random.choice(self.valid_idx)
        self.digit, _ = self.mnist[idx]
        self.state = np.clip( (np.pi/2)*np.random.randn(), -np.pi/2, np.pi/2)
        
        return self._get_obs()
        
    def step(self, action):
        # Execute one time step within the environment
        self.state += action / 10.
        
        done = np.abs(self.state) >  3*np.pi/2
        
        reward = -self.state**2
        
        return self._get_obs(), reward, done, {}
        
    def _get_obs(self):
        # returns image corresponding to current state
        image = TF.rotate(self.digit, self.state*180/np.pi)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.1307,), (0.3081,))(image)
        return image
    
    
class StatePIDPolicy:
    def __init__(self, kp=1.):
        self.kp = kp
        self.reference = 0
    
    def act(self, state):
        err = state - self.reference
        action = -self.kp*err
        return action
    
    
class SafePolicy:
    def __init__(self):
        pass
    
    def act(self, state):
        action = 0
        return action
    
    