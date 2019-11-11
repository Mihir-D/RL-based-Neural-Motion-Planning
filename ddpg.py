import torch
import os
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from replay_buffer import ReplayBuffer
from actor import Actor
from critic import Critic


class DDPGAgent:
    def __init__(self, env, config, environment_config):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.environment_config = environment_config
        self.config = config
        self.env = env
        self.dim_states = env.robot.GetDOF() + self.environment_config['w']['output_size']
        self.dim_actions = env.robot.GetDOF() - 1
        self.gamma = self.config['model']['gamma']
        self.tau = self.config['model']['tau']
        self.save_dir = self.config['model']['save_dir']
        self.batch_size = self.config['model']['batch_size']

        self.device = torch.device('cuda')
       
        self.actor = Actor(self.dim_states, self.config['model']['actor']['hidden_layers'], self.dim_actions).to(self.device)
        self.actor_target = Actor(self.dim_states, self.config['model']['actor']['hidden_layers'], self.dim_actions).to(self.device)
        self.actor_target.eval()
        self.critic = Critic(self.dim_states + self.dim_actions, self.config['model']['critic']['hidden_layers'], 1).to(self.device)
        self.critic_target = Critic(self.dim_states + self.dim_actions, self.config['model']['critic']['hidden_layers'], 1).to(self.device)
        self.critic_target.eval()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
          
        self.replay_buffer = ReplayBuffer(config['model']['replay_buffer_size'])        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr = self.config['model']['actor']['learning_rate'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.config['model']['critic']['learning_rate'])

        self.save_dir = self.config['model']['save_dir']
    
    def get_action(self, state, workspace_features):
        state = torch.from_numpy(state).float().squeeze(0).to(self.device)
        combined_state = torch.cat([state, workspace_features], 0)
        action = self.actor.forward(combined_state)
        action = action.detach().cpu().clone().numpy()
        action = action * self.environment_config['path']['action_step_size']
        return action

    def process_states_to_train(self, states):
        processed_states = [state.detach().cpu().clone().numpy() for state in states]
        return Variable(torch.cuda.FloatTensor(processed_states), requires_grad = True)
    
    def update(self):
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(self.batch_size)
        # states = Variable(torch.cuda.FloatTensor(states), requires_grad = True)
        states_train = self.process_states_to_train(states)
        actions = torch.cuda.FloatTensor(actions)
        rewards = torch.cuda.FloatTensor(rewards).view(-1, 1)
        next_states_train = self.process_states_to_train(next_states)
        # next_states = Variable(torch.cuda.FloatTensor(next_states), requires_grad = True)
     
        Q_vals = self.critic.forward(states_train, actions)
        next_actions = self.actor_target.forward(next_states_train)
        Q_next = self.critic_target.forward(next_states_train, next_actions.detach())
        Q_target = rewards + self.gamma * Q_next
        critic_loss = self.critic_criterion(Q_vals, Q_target)

        policy_loss = -self.critic.forward(states_train, self.actor.forward(states_train)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()        

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.env.workspace_features_optimizer.zero_grad()
        for i, state in enumerate(states):
            loss_criterion = nn.MSELoss()
            loss = loss_criterion(state, states_train[i] + states_train.grad[i])
            loss.backward(retain_graph = True)
        self.env.workspace_features_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save_model(self):
        actor_save_path = os.path.join(os.getcwd(), self.save_dir, 'actor_openrave.pth')
        torch.save(self.actor.state_dict(), actor_save_path)
        critic_save_path = os.path.join(os.getcwd(), self.save_dir, 'critic_openrave.pth')
        torch.save(self.critic.state_dict(), critic_save_path)
    
    def load_model(self):
        actor_save_path = os.path.join(os.getcwd(), self.save_dir, 'actor_openrave.pth')
        self.actor.load_state_dict(torch.load(actor_save_path))
        critic_save_path = os.path.join(os.getcwd(), self.save_dir, 'critic_openrave.pth')
        self.critic.load_state_dict(torch.load(critic_save_path))

        self.actor.eval()
        self.critic.eval()