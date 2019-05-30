# deep-reinforcement-learning-project1
## 1. The Environment
The target of the project is to train an agent to navigate around a square world and collect yellow bananas as many as possible while avoiding the blue ones.

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
