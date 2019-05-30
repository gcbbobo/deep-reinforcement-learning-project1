import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        '''
        print('------------------------------------------------------')
        print('Q_targets_next:')
        print('self.qnetwork_target(next_states)')
        # tensor([[# of actions],[],...,[]]): 64 x 4 
        # accumulate at least a batch size of <states, actions, rewards, next_states, dones> data before learning
        
        
        print(self.qnetwork_target(next_states))
        print('self.qnetwork_target(next_states).detach()')        
        print(self.qnetwork_target(next_states).detach())
        # detach just for avoid being processed by backward
        
        print('self.qnetwork_target(next_states).detach().max(1)')
        print(self.qnetwork_target(next_states).detach().max(1))
        
        # (tensor([0.1493, 0.3575, 0.2237, 0.1022, 0.1110, 0.0570, 0.0760, 0.3279, 0.1853,
        # 0.1074, 0.1031, 0.0671, 0.0762, 0.0975, 0.2320, 0.1199, 0.2608, 0.0442,
        # 0.0899, 0.1242, 0.1847, 0.1406, 0.2220, 0.1987, 0.1166, 0.3334, 0.1095,
        # 0.1296, 0.0902, 0.2691, 0.1167, 0.0830, 0.0564, 0.0783, 0.1181, 0.1657,
        # 0.0571, 0.0430, 0.1595, 0.1537, 0.1266, 0.1955, 0.1512, 0.1130, 0.0734,
        # 0.1166, 0.2058, 0.0394, 0.1984, 0.1287, 0.0788, 0.1323, 0.0728, 0.1063,
        # 0.1980, 0.0997, 0.1367, 0.1249, 0.1767, 0.0927, 0.0816, 0.0921, 0.0378,
        # 0.1898], device='cuda:0'), tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        # 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
        # 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1], device='cuda:0'))
        # The first set of tensor is the value-function, the second set of tensor is the action index taken
        
        
        print('self.qnetwork_target(next_states).detach().max(1)[0]')
        print(self.qnetwork_target(next_states).detach().max(1)[0])
        # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
        # The maximum value of each row--> pick the value function - action pair with the largest probability
        
        print('Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)')
        print(self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1))
        # adding one more dimension
        print('')
        '''
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        
        '''
        # gather is like index searching.
        
        =======
        torch.gather(input, dim, index, out=None) → Tensor

        Gathers values along an axis specified by dim.

        For a 3-D tensor the output is specified by:

        out[i][j][k] = input[index[i][j][k]][j][k]  # dim=0
        out[i][j][k] = input[i][index[i][j][k]][k]  # dim=1
        out[i][j][k] = input[i][j][index[i][j][k]]  # dim=2

        Parameters: 

            input (Tensor) – The source tensor
            dim (int) – The axis along which to index
            index (LongTensor) – The indices of elements to gather
            out (Tensor, optional) – Destination tensor

        Example:

        >>> t = torch.Tensor([[1,2],[3,4]])
        >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
         1  1
         4  3
        [torch.FloatTensor of size 2x2]
        
        =====
        Explanation:
        For each element in the input matrix, its innate index is born(index given): 
        value 0 at 0 row, 0 col: i = 0, j = 0
        dim = 1 : The search is implemented along axis = 1, then fixed the row index:i = 0
        value 0 is the row index to search(index to search)
        Then: gather[0][0] = t[row index fixed:i = 0][col index to search: value = 0] = 1 
        Similarly, for:
        value 1 at 1 row, 0 col: i = 1, j = 0
        gather[1][0] = t[row index fixed:i = 1][col index to search: value = 1] = 4
        
        '''
        
        
        # action is the index to trace
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        '''
        print('------------------------------------------------------')
        print('Q_expected:')
        print('self.qnetwork_local(states)')
        print(self.qnetwork_local(states))
        print(' Q_expected = self.qnetwork_local(states).gather(1, actions)')
        # self.qnetwork_local(states): 64 x 4
        
        print(self.qnetwork_local(states).gather(1, actions))
        print('')
        '''

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)