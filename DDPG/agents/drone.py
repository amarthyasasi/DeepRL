from torch.serialization import storage_to_tensor_type
from sensors.heat_sensor import HeatSensor
import torch
import airsim
from collections import deque, namedtuple
import numpy as np
import time
from PIL import Image

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class OUNoise:
    '''
    Noise model for RL agent
    '''
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):

        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):

        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx

        return self.state


class Drone:
    '''
    Drone class
    '''
    def __init__(self, start_position, velocity_factor, hparams, buffer, sensor):
        '''
        Drone initializations
        start position, velocity factor, hyper parameters, replay buffer
        '''
        self.buffer = buffer
        self.sensor = sensor
        self.hparams = hparams
        self.start_position = start_position
        self.scaling_factor = velocity_factor
        self.client = None
        self.noise = OUNoise(action_dimension = 3)
        self.num_batch = 3
        self.reset()
        self.previous_states = deque(maxlen=self.num_batch)

    def initializeClient(self):
        '''
        Initializing airsim client
        '''

        if self.client is None:
            self.client = airsim.MultirotorClient(**self.hparams.environment.server)
            self.client.confirmConnection()
        else:
            self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def hasCollided(self):
        '''
        Check if Drone has collided
        '''

        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        else:
            return False

    def batchStates(self, exp):
        '''
        Batch state dictionaries into one
        '''

        states, actions, rewards, dones, next_states = exp

        self.previous_states.append(exp)
        len_buffer = len(self.previous_states)
        #print("action in agent", actions.shape)

        if len_buffer < self.num_batch:
            image_zero = torch.zeros_like(states["image"])
            signal_zero = torch.zeros_like(states["signal"])
            action_zero = torch.zeros_like(actions)
            rewards_zero = torch.zeros_like(torch.tensor([rewards]))
            dones_zero = torch.zeros_like(torch.tensor([dones]))

        image_batch = []
        signal_batch = []
        new_signal_batch = []
        new_image_batch = []

        for i in range(self.num_batch):

            if i < self.num_batch - len_buffer:
                #print("zeros")
                image_batch.append(image_zero)
                signal_batch.append(signal_zero)
                new_image_batch.append(image_zero)
                new_signal_batch.append(signal_zero)
            else:
                index = i - (self.num_batch - len_buffer)
                exp_old = self.previous_states[index]

                states_old, actions_old, rewards_old, dones_old, next_states_old = exp_old

                image_batch.append(states_old["image"])
                signal_batch.append(states_old["signal"])
                new_image_batch.append(next_states_old["image"])
                new_signal_batch.append(next_states_old["signal"])

        state_out = {"image": torch.cat(image_batch, dim = 0), "signal":torch.cat(signal_batch, dim = 0)}

        next_state_out = {"image": torch.cat(new_image_batch, dim = 0), "signal": torch.cat(new_signal_batch, dim = 0)}

        exp_out = Experience(state=state_out, action = actions, reward = torch.tensor([rewards]), done = torch.tensor([dones]), new_state = next_state_out)

        return exp_out

    def hasReachedGoal(self):
        '''
        Check if Drone has reached the goal (distance is less than the velocity factor)
        '''

        current_position = self.convertPositionToTensor(self.position.position)
        distance = self.sensor.getDistanceFromDestination(current_position)
        if distance < self.hparams.model.thresh_dist:
            print("#"*100)
            print("Reached goal")
            return True
        else:
            return False

    def nextAction(self, action):
        '''
        Get change of position from action index
        '''

        scaling_factor = self.scaling_factor
        if action == 0:
            quad_offset = (scaling_factor, 0, 0)

        elif action == 1:
            quad_offset = (0, scaling_factor, 0)

        elif action == 2:
            quad_offset = (0, 0, scaling_factor)

        elif action == 3:
            quad_offset = (-scaling_factor, 0, 0)

        elif action == 4:
            quad_offset = (0, -scaling_factor, 0)

        elif action == 5:
            quad_offset = (0, 0, -scaling_factor)

        else:
            quad_offset = (0, 0, 0)

        return quad_offset


    def convertPositionToTensor(self, position):
        '''
        Converts position from airsim vector 3d to 3 x 1 tensor
        '''

        current_position = torch.tensor([[position.x_val], [position.y_val], [position.z_val]])

        return current_position


    def getAgentState(self):
        '''
        Get agent state (Image and signal strength)
        '''

        position = self.convertPositionToTensor(self.position.position)
        state_image = self.getImage()
        state_signal_strength = self.sensor.getSignalStrength(position)

        state_image = torch.tensor(state_image).permute(2, 0, 1).float()
        state_signal_strength = torch.tensor([state_signal_strength]).float()

        #print(state_image.max())
        state_image = state_image / 255.0

        #print(state_image.shape, state_signal_strength.shape)
        return {"image": state_image, "signal": state_signal_strength}

    def getAction(self, net, epsilon, device):
        '''
        Perform action
        '''
        state_dict = self.getAgentState()
        exp_out = Experience(state=state_dict, action = state_dict["signal"], reward = state_dict["signal"], done = state_dict["signal"], new_state = state_dict)
        exp = self.batchStates(exp_out)
        state_dict, _, _, _, _ = exp

        if device not in ['cpu']:
            for key in state_dict:
                state_dict[key] = state_dict[key].cuda(device)

        #print(state_dict["image"].shape,state_dict["signal"].shape )
        action = net(state_dict["image"].unsqueeze(0), state_dict["signal"].unsqueeze(0)).detach().cpu().numpy().squeeze()
        if np.random.random() < epsilon:
            action_out = torch.tensor([action[0], action[1], action[2]])
            noise_out = self.noise.noise()
            action = torch.tensor([np.clip(action_out[0] + noise_out[0], -1, 1), np.clip(action_out[1] + noise_out[1], -1, 1), np.clip(action_out[2] + noise_out[2], -1, 1)])
        else:
            action = torch.tensor([action[0], action[1], action[2]])

        return action


    @torch.no_grad()
    def playStep(self, net, epsilon, device):
        '''
        Performs one step in the environment

        Input:
            net - DQN network
            device - device

        Returns:
            reward - instantaneous reward
            done - Check if episode is done
        '''


        action_offset = self.getAction(net, epsilon, device)
        # action_offset = self.nextAction(action)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        state_dict = self.getAgentState()
        #print(action_offset, quad_state)
        self.client.moveByVelocityAsync(
            quad_vel.x_val + action_offset[0].numpy(),
            quad_vel.y_val + action_offset[1].numpy(),
            quad_vel.z_val + action_offset[2].numpy(),
            0.5,
        ).join()
        #time.sleep(0.5)

        # print(self.position)
        current_position = self.convertPositionToTensor(self.position.position)
        done, reward = self.isDone()

        new_state_dict = self.getAgentState()
        self.position = self.client.simGetVehiclePose()
        print(self.position.position)
        # print(self.client.getMultirotorState().kinematics_estimated.position)
        if not done:
            reward = self.sensor.getReward(current_position)
        print(reward)

        # exp = batchStates(state_dict, action_offset, reward, done, new_state_dict)
        # exp = Experience(state_dict, action_offset, reward, done, new_state_dict)
        exp_out = Experience(state_dict, action_offset, reward, done, new_state_dict)
        exp = self.batchStates(exp_out)

        self.buffer.append(exp)

        if done:
            self.reset()

        return reward, done

    def isDone(self):
        '''
        Check if the drone has reached goal or collided
        '''

        reward = 0

        # Either reached goal or collided
        if self.hasReachedGoal():
            reward = self.hparams.environment.reward.goal
            done = 1
        elif self.hasCollided():
            reward = self.hparams.environment.reward.collision
            done = 1
        else:
            done = 0

        return done, reward


    def postprocessImage(self, responses):
        '''
        Process image from airsim responses
        '''

        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgba = img1d.reshape(response.height, response.width, 3)
        img2d = np.flipud(img_rgba)

        #print(img2d.shape)
        #image = Image.fromarray(img2d)
        #image_out = np.array(image.convert("L"))

        image_out = img2d.copy()
        return image_out

    def reset(self):
        '''
        Reset Drone to start position at the end of an episode
        '''

        self.initializeClient()
        self.position = self.client.simGetVehiclePose()

        self.previous_states = deque(maxlen=self.num_batch)

        pose_params = self.hparams.environment
        start = self.start_position.numpy()

        # Set initial pose
        self.position.position.x_val = float(start[0, 0])
        self.position.position.y_val = float(start[1, 0])
        self.position.position.z_val = float(start[2, 0])

        self.position.orientation.w_val = pose_params.quaternion.start.w_val
        self.position.orientation.x_val = pose_params.quaternion.start.x_val
        self.position.orientation.y_val = pose_params.quaternion.start.y_val
        self.position.orientation.z_val = pose_params.quaternion.start.z_val

        # print("position, ", self.position)
        # Set init position
        self.client.moveToPositionAsync(float(start[0, 0]),float(start[1, 0]),float(start[2, 0]), 10).join()
        self.client.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        #
        #self.client.simSetVehiclePose(self.position, True)
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        #print(self.state)

        # Take off with the drone
        #self.client.takeoffAsync().join()
        #time.sleep(0.5)
        self.position = self.client.simGetVehiclePose()
        print(self.state, self.position)
        print("#" * 30)
        print("RESET")

    def getImage(self):
        '''
        Get observation from drone sensors
        '''
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        return self.postprocessImage(responses)


if __name__=="__main__":


    agent = Drone(5)

