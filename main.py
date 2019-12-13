import sys
import os
import yaml
# import gym
import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPGAgent
from ou_noise import OUNoise
from openrave_manager import OpenRaveManager
import tensorflow as tf
import argparse
import time
# from gaussian_noise import GaussianNoise

def parse():
    parser = argparse.ArgumentParser(description = 'ddpg')
    parser.add_argument('--train_ddpg', action='store_true', help='whether train DDPG')
    parser.add_argument('--test_ddpg', action='store_true', help='whether test DDPG')

    args = parser.parse_args()
    return args

class DDPG(object):
    def __init__(self, config, environment_config):
        # self.device = torch.device('cuda')
        self.environment_config = environment_config
        self.config = config
        self.env = OpenRaveManager(self.environment_config)
        self.agent = DDPGAgent(self.env, self.config, self.environment_config)
        # self.noise = GaussianNoise(self.env.get_joint_bounds())
        self.batch_size = self.config['model']['batch_size']
        self.num_episodes = self.config['model']['num_episodes']
        self.steps_per_episode = self.config['model']['steps_per_episode']
        self.rewards = []
        self.avg_rewards = []

    def get_combined_state(self, state, workspace_features):   
        combined_state = np.hstack([state, workspace_features])
        return combined_state
    
    def run_episode(self, episode):
        state, workspace_features = self.env.reset()
        # self.noise.reset()
        episode_reward = 0
        print("\n")
        print(workspace_features)
        for step in range(self.steps_per_episode):
            combined_state = self.get_combined_state(state, workspace_features)
            action = self.agent.noise_action(combined_state)
            next_state, reward, done = self.env.step(action) 

            combined_next_state = self.get_combined_state(next_state, workspace_features)

            self.agent.perceive(combined_state,action,reward,combined_next_state,done)        

            state = next_state
            episode_reward += reward

            if done:
                sys.stdout.write("episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2), 
                        np.mean(self.rewards[-10:])))
                self.rewards.append(episode_reward)
                self.avg_rewards.append(np.mean(self.rewards[-10:]))
                sys.stdout.write("State: {} \n".format(state))
                return
        sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), 
                        np.mean(self.rewards[-10:])))
        self.rewards.append(episode_reward)
        self.avg_rewards.append(np.mean(self.rewards[-10:]))
        sys.stdout.write("State: {} \n".format(state))

    
    def run_test_episode(self):
        state, workspace_features = self.env.reset(test = True)
        episode_reward = 0

        print('\n')
        print('Robot configuration at the start of the episode : ')
        print(state)
        for step in range(self.steps_per_episode):
            time.sleep(1)
            combined_state = self.get_combined_state(state, workspace_features)
            action = self.agent.action(combined_state)
            next_state, reward, done = self.env.step(action)     
            state = next_state
            episode_reward += reward
            # print(action, reward)
            if done:
                if reward > 0:
                    self.successful_test_episodes += 1
                print('Robot configuration at end of the episode : ')
                print(state)
                print('Desired goal configuration : ')
                print(self.env.robot_goal_configuration)
                sys.stdout.write("episode reward: {} \n".format(episode_reward))
                return
        print('Robot configuration at end of the episode : ')
        print(state)
        print('Desired goal configuration : ')
        print(self.env.robot_goal_configuration)
        sys.stdout.write("episode reward: {} \n".format(episode_reward))

    def train(self):
        for episode in range(self.num_episodes):
            self.run_episode(episode)
        self.agent.actor_network.save_network()
        self.agent.critic_network.save_network()

    def test(self):
        self.successful_test_episodes = 0
        for episode in range(self.config['test']['num_episodes']):
            self.run_test_episode()
            time.sleep(2)
        print('Number of successful episodes : ' +  str(self.successful_test_episodes))
        print('Success percentage : ' + str(self.successful_test_episodes / float(self.config['test']['num_episodes'])))
    
    def plot(self):
        plt.plot(self.rewards)
        plt.plot(self.avg_rewards)
        plt.plot()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
    
if __name__ == '__main__':

    config_path = os.path.join(os.getcwd(), 'config', 'config.yml')
    with open(config_path, 'r') as yaml_file:
        config = yaml.load(yaml_file)
        print(yaml.dump(config))
    
    config_path = os.path.join(os.getcwd(), 'config', 'environment_config.yml')
    with open(config_path, 'r') as yaml_file:
        environment_config = yaml.load(yaml_file)
        print(yaml.dump(environment_config))
    

    ddpg = DDPG(config, environment_config)
    args = parse()
    if args.train_ddpg:
        ddpg.train()
        ddpg.plot()
    if args.test_ddpg:
        time.sleep(10)
        ddpg.test()