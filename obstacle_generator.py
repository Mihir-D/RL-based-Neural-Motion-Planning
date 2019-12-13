import openravepy
import numpy as np


class Obstacle(object):
    def __init__(self, side_x, side_z, center_x, center_z, theta_y):
        self.side_x = side_x
        self.side_z = side_z
        self.center_x = center_x
        self.center_z = center_z
        self.theta_y = theta_y

class ObstacleGenerator(object):
    def __init__(self, config):
        self.config = config
        self.max_obstacles = self.config['obstacles']['max_obstacles']
        self.min_obstacles = self.config['obstacles']['min_obstacles']
        self.obstacle_probabilities = self.config['obstacles']['obstacle_probabilities']

        self.x_max_center = self.config['obstacles']['x_max_center']
        self.x_min_center = self.config['obstacles']['x_min_center']
        self.z_max_center = self.config['obstacles']['z_max_center']
        self.z_min_center = self.config['obstacles']['z_min_center']
        
        self.max_side_length = self.config['obstacles']['max_side_length']
        self.min_side_length = self.config['obstacles']['min_side_length']
    
    def generate_single_obstacle(self):
        center_x = np.random.uniform(self.x_min_center, self.x_max_center)
        center_z = np.random.uniform(self.z_min_center, self.z_max_center)
        side_x = np.random.uniform(self.min_side_length, self.max_side_length)
        side_z = np.random.uniform(self.min_side_length, self.max_side_length)
        theta_y = np.random.uniform(0, np.pi / 2)
        
        ray_angle = np.random.uniform(0, np.pi)
        rotation_matrix = np.array([[np.cos(ray_angle), -np.sin(ray_angle)], [np.sin(ray_angle), np.cos(ray_angle)]])
        center = np.array([[center_x], [center_z]])
        center = np.dot(rotation_matrix, center)
        center_x = center[0, 0]
        center_z = center[1, 0]

        return Obstacle(side_x, side_z, center_x, center_z, theta_y)
    
    def generate_obstacles(self):
        """
            Generates obstacles and returns the length along x, z axis, center in x, z axis and theta about y axis for each obstacle
        """
        num_obstacles = np.random.choice(np.arange(self.min_obstacles, self.max_obstacles + 1), p = self.obstacle_probabilities)
        obstacles = []
        for obstacle in range(num_obstacles):
            obstacles.append(self.generate_single_obstacle())
        return obstacles