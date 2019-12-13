import openravepy
import os
import yaml
import numpy as np
import torch
import random
import torch.optim as optim
from obstacle_generator import ObstacleGenerator
from workspace_feature_vector import WorkspaceFeature

class OpenRaveManager(object):
    def __init__(self, config):
        self.device = torch.device('cuda')

        self.config = config
        self.obstacle_generator = ObstacleGenerator(self.config)
        env_path = os.path.join(os.getcwd(), 'config', 'widowx_env.xml')
        self.env = openravepy.Environment()
        self.env.SetViewer('qtcoin')
        self.env.StopSimulation()
        self.env.Load(env_path)
        self.robot = self.env.GetRobots()[0]
        self.obstacles = [] # objects of type KinBody 
        self.obstacles_data = [] # objects of type #Obstacle defined in obstacle_generator.py
        self.max_attempts_to_initial_configuration = self.config['obstacles']['max_attempts_to_initial_configuration']
        self.initialize_robot_position() # Get random goal configuration
        self.robot_goal_configuration = self.robot.GetDOFValues()

    def reset(self):
        self.delete_obstacles()
        self.add_obstacles()
        self.initialize_robot_position()
        self.robot_start_configuration = self.robot.GetDOFValues()
        self.workspace_features = self.get_workspace_features()
        
        return self.robot_start_configuration, self.workspace_features
    
    def partition_segment(self, start_configuration, action):
        start_configuration = np.array(start_configuration)
        action = np.array(action)

        distance_norm = np.linalg.norm(action)
        step_size = self.config['path']['segment_step_size']
        
        if distance_norm < step_size:
            return [start_configuration + action]
        else:
            step = action * (step_size / distance_norm)
            steps = []
            current_configuration = np.copy(start_configuration)
            for _ in range(int(distance_norm / step_size)):
                current_configuration = current_configuration + step
                steps.append(current_configuration)
            steps.append(start_configuration + action)
            return steps
        
    def check_segment_validity(self, start_configuration, action):
        steps = self.partition_segment(start_configuration, action)
        for step in steps:
            step = step[1:] # since valid configuration function takes only 4 joint values
            if not self.is_valid_configuration(step):
                return False
        return True
    
    def is_near_goal(self, current_configuration):
        current_configuration = np.array(current_configuration)
        goal_configuration = np.array(self.robot_goal_configuration)
        distance = np.linalg.norm(current_configuration - goal_configuration)
        return distance < self.config['path']['goal_distance']

    def truncate_actions(self, current_configuration, action):
        joint_bounds = self.get_joint_bounds()
        current_configuration = np.array(current_configuration)
        action = np.array(action)
        end_configuration = current_configuration + action
        truncated_configuration = np.copy(end_configuration)

        for i in range(self.robot.GetDOF()):
            truncated_configuration[i] = max(end_configuration[i], joint_bounds[0][i])
            truncated_configuration[i] = min(truncated_configuration[i], joint_bounds[1][i])

        distance = np.linalg.norm(truncated_configuration - end_configuration)
        return distance, truncated_configuration - current_configuration


    def step(self, action):
        current_robot_configuration = self.robot.GetDOFValues() # robot configuration is current position of all 5 joints 
        action = np.pad(action, (1, 0), 'constant', constant_values = (0,))  # action to be taken is for 4 joints only since first joint is always disabled
        reward = 0.0
        distance, action = self.truncate_actions(current_robot_configuration, action)
        reward -= distance * self.config['reward']['truncate_penalty']

        if self.check_segment_validity(current_robot_configuration, action):
            if self.is_near_goal(current_robot_configuration):
                reward += 1.0
                return self.robot.GetDOFValues(), reward, True
            reward -= self.config['reward']['keep_alive_penalty']
            return self.robot.GetDOFValues(), reward, False
        else:
            self.robot.SetDOFValues(current_robot_configuration)
            reward -= 1.0
            return current_robot_configuration, reward, True

    def get_goal_reward(self):
        return 1
    
    def delete_obstacles(self):
        for obstacle in self.obstacles:
            self.env.Remove(obstacle)
        self.obstacles = []

    def add_obstacles(self):
        side_y = self.config['obstacles']['side_y']
        self.obstacles_data = self.obstacle_generator.generate_obstacles()        
        for i in range(len(self.obstacles_data)):
            obstacle = self.obstacles_data[i]
            body = openravepy.RaveCreateKinBody(self.env, '')
            body.SetName('box{}'.format(i))
            body.InitFromBoxes(np.array([[0, 0, 0, obstacle.side_x, side_y, obstacle.side_z]]),
                                True)
            self.env.Add(body, True)

            transformation_matrix = np.eye(4)
            translation = np.array([
                obstacle.center_x, 0.0, obstacle.center_z])

            theta = obstacle.theta_y
            rotation_matrix = np.array([
                [np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]
            ])
            transformation_matrix[:3, -1] = translation
            transformation_matrix[:3, :3] = rotation_matrix
            body.SetTransform(transformation_matrix)
            self.obstacles.append(body)

    def is_valid_configuration(self, joints):
        self.robot.SetDOFValues(joints, [1, 2, 3, 4])
        valid = not self.robot.CheckSelfCollision()
        for obstacle in self.obstacles:
            valid = valid and not self.env.CheckCollision(self.robot, obstacle)
        return valid
    
    def get_joint_bounds(self):
        return self.robot.GetDOFLimits()

    def initialize_robot_position(self):
        self.robot.SetActiveDOFs(range(1, 5))
        valid_configuration = False
        joint_bounds = self.get_joint_bounds()
        attempts = 0
        while not valid_configuration:
            attempts += 1
            joint_values = []
            for i in range(1, self.robot.GetDOF()):
                joint_values.append(np.random.uniform(joint_bounds[0][i], joint_bounds[1][i]))
            valid_configuration = self.is_valid_configuration(joint_values)

            if attempts == self.max_attempts_to_initial_configuration:
                self.delete_obstacles()
                self.add_obstacles()
                attempts = 0
            

    def get_workspace_features(self):
        # self.initialize_robot_position() # Get random goal configuration
        # self.robot_goal_configuration = self.robot.GetDOFValues()
        self.robot.SetDOFValues(self.robot_start_configuration) # reset robot back to initial configuration

        if len(self.obstacles_data) > 0:
            obstacles_data = np.hstack((obstacle_data.side_x, obstacle_data.side_z, obstacle_data.center_x, obstacle_data.center_z, 
                obstacle_data.theta_y) for obstacle_data in self.obstacles_data)
        else:
            obstacles_data = np.array([])
        workspace_features = np.hstack([self.robot_goal_configuration, obstacles_data])
        workspace_features = np.pad(workspace_features, (0, self.robot.GetDOF() + 5 * self.config['obstacles']['max_obstacles'] - workspace_features.shape[0]), 'constant', constant_values = (0,))
        
        return workspace_features