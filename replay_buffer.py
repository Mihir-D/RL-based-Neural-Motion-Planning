from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        batch = random.sample(self.buffer, batch_size)

        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(done)
        
        return states, actions, rewards, next_states, terminals

    def __len__(self):
        return len(self.buffer)