import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.advantage = nn.Linear(fc2_units, action_size)
        # For a given state (combining a lot of physical states), the size of v(s) = 1
        self.statevalue = nn.Linear(fc2_units, 1)
        #self.fc3 = nn.Linear(1+action_size, action_size)
        
        self.state_size = state_size
        self.action_size = action_size

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        V = F.relu(self.statevalue(x))
        adv = F.relu(self.advantage(x))
        
        # transfer the scalar V to matrix V for adding to the advantage function
        V = V.expand(-1,self.action_size)
        
        '''
        print('------')
        print('state')
        print(state)
        # state is in form of:[[]]
        
        print('adv')
        print(adv)
        print('adv.mean(1)')
        print(adv.mean(1))
        
        
        print('adv.mean(1).unsqueeze(1)')
        print(adv.mean(1).unsqueeze(1))
        '''
        
        '''
        adv
        tensor([[0.0420, 0.1167, 0.0000, 0.0000]], device='cuda:0')
        adv.mean(1)
        tensor([0.0397], device='cuda:0')
        '''
        # That's why we need unsqueeze to adding up one dimension since mean reduce 1 dimension during the process
        x = V + adv - adv.mean(1).unsqueeze(1).expand(-1,self.action_size)
        
        #x = self.fc3(x)
        
        return x
