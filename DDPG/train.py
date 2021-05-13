import torch
from env import Drone
import os, hydra, logging, glob
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from ActorCritic import Actor, Critic
from buffers import *
import torch.nn as nn
from collections import OrderedDict

class AgentTrainer(pl.LightningModule):
    '''
    Pytorch trainer class for Drone Reinforcement learning
    '''

    def __init__(self, hparams):
        '''
        Initializations
        '''
        super().__init__()
        self.hparams = hparams

        # Position of human
        source_position = torch.tensor([[self.hparams.environment.position.end.x],
                                        [self.hparams.environment.position.end.y],
                                        [self.hparams.environment.position.end.z]]).float()

        # Position of agent
        agent_position  = torch.tensor([[self.hparams.environment.position.start.x],
                                        [self.hparams.environment.position.start.y],
                                        [self.hparams.environment.position.start.z]]).float()


        # Initialize Replay buffer
        self.replay_buffer = ReplayBuffer(capacity = self.hparams.model.replay_buffer_size)


        # Initialize drone
        self.agent = Drone(start_position = agent_position,
                           goal_position = source_position,
                           velocity_factor = self.hparams.environment.agent.velocity_factor,
                           hparams = self.hparams,
                           buffer = self.replay_buffer)

        # Actor networks
        self.net = Actor(**self.hparams.model.actor)
        self.target_net = Actor(**self.hparams.model.actor)

        # Critic networks
        self.critic = Critic(**self.hparams.model.critic)
        self.target_critic = Critic(**self.hparams.model.critic)

        # Hard update
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.total_reward = -10000
        self.episode_steps = 0.0
        self.max_episode_steps = self.hparams.model.max_episode
        self.episode_reward = 0.0
        self.populate(self.hparams.model.replay_buffer_size)


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def configure_optimizers(self):

        optimizer2 = getattr(torch.optim, self.hparams.optimizer.type)([{"params": self.net.parameters(), "lr": self.hparams.optimizer.args.lr}], **self.hparams.optimizer.args)
        optimizer = getattr(torch.optim, self.hparams.optimizer.type)(self.critic.parameters(), **self.hparams.optimizer.args, weight_decay=1e-3)

        scheduler2 = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.type)(optimizer, **self.hparams.scheduler.args)
        scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.type)(optimizer, **self.hparams.scheduler.args)

        return [optimizer, optimizer2], [scheduler, scheduler2]

    def dqn_mse_loss(self, batch) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        #print(states["image"].shape, rewards.shape)
        rewards_out = rewards[:, -1]
        print(actions.shape, rewards_out.shape, rewards.shape, "shapes")
        #print(rewards.shape, actions.shape, "reward, action")
        # print(states["image"].shape)
        # state_action_values = self.net(states["image"], states["signal"]).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        action_value = self.net(next_states["image"])
        Q_value = self.critic(next_states["image"], action_value).squeeze(-1)

        # print(state_action_values)

        with torch.no_grad():


            #next_action_value = self.target_net(next_states["image"], next_states["signal"])
            #print(next_action_value.shape, "action")
            next_Q_value = self.target_critic(states["image"], actions.float()).squeeze(-1)
            # next_state_values[dones] = 0.0
            #print("Q value:", next_Q_value.shape)
            #next_action_value = next_action_value.detach()
            next_Q_value = next_Q_value.detach()

            #Q_value_actor = self.critic(next_states["image"], next_states["signal"], action_value).squeeze(-1)

        #print(next_Q_value.shape, rewards_out.shape)
        expected_state_action_values = Q_value * self.hparams.model.gamma + rewards_out
        #print(expected_state_action_values.shape, Q_value.shape)
        return {"loss": nn.MSELoss()(next_Q_value, expected_state_action_values), "policy_loss": - (Q_value).mean()}

    def populate(self, steps: int = 1000) -> None:
        '''
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        '''

        for i in range(steps):
            print(i)
            self.agent.playStep(self.net, 1.0, self.get_device())

            if i % self.max_episode_steps == 0:
                self.agent.reset()

        self.agent.reset()

    def playTrajectory(self):
        '''
        Play the trajectory
        '''
        self.agent.reset()
        device = self.get_device()
        while (True):

            self.agent.playStep(self.net, 0, device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        '''
        Training steps
        '''

        self.episode_steps = self.episode_steps + 1
        device = self.get_device()
        epsilon = max(self.hparams.model.min_epsilon, self.hparams.model.max_epsilon - (self.global_step + 1) / self.hparams.model.stop_decay)
        print("eps:", epsilon)

        # step through environment with agent
        reward, done = self.agent.playStep(self.target_net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)
        #print(loss)
        self.log("train_loss", loss["loss"], on_epoch = True, prog_bar = True, on_step = True, logger = True)
        self.log("policy_loss", loss["policy_loss"], on_epoch = True, prog_bar = True, on_step = True, logger = True)

        if done:
            if self.episode_reward > self.total_reward:
                self.total_reward = self.episode_reward

            self.episode_reward = 0
            self.episode_steps = 0


        if optimizer_idx:
            loss_out = loss["policy_loss"]
        else:
            loss_out = loss["loss"]

        # Soft update of target network
        if self.global_step % self.hparams.model.sync_rate == 0:

            self.soft_update(self.target_net, self.net, self.hparams.model.tau)
            self.soft_update(self.target_critic, self.critic, self.hparams.model.tau)

            # self.target_net.load_state_dict(self.net.state_dict())
            # self.target_critic.load_state_dict(self.critic.state_dict())

        log = {
            'total_reward': torch.tensor(self.total_reward).to(device),
            'reward': torch.tensor(reward).to(device),
            'steps': torch.tensor(self.global_step).to(device)
        }
        for key in log:
            self.log(key, log[key], logger = True, prog_bar = True, on_step = True)

        if self.episode_steps > self.max_episode_steps:
            self.episode_steps = 0
            self.total_reward = self.episode_reward
            self.agent.reset()
        #print(loss_out)
        #return OrderedDict({'loss': loss, 'log': log, 'progress_bar': log})
        return loss_out


    def __dataloader(self) -> DataLoader:
        """
        Initialize the Replay Buffer dataset used for retrieving experiences
        """

        dataset = RLDataset(self.replay_buffer, self.hparams.model.sample_size)
        dataloader = DataLoader(
            dataset=dataset,
            **self.hparams.dataset.loader)

        return dataloader

    def train_dataloader(self) -> DataLoader:
        """
        Get train loader
        """

        return self.__dataloader()

    def get_device(self) -> str:
        """
        Retrieve device currently being used by minibatch
        """

        return self.device.index if self.on_gpu else 'cpu'

    def forward(self, x):

        return self.net(x)


seed_everything(123)
log = logging.getLogger(__name__)

@hydra.main(config_path="./", config_name="HyperParams.yaml")
def main(cfg):

    tb_logger = TensorBoardLogger(save_dir = "./")
    log.info(cfg.pretty())
    model = AgentTrainer(hparams = cfg)
    trainer = Trainer(**cfg.trainer, logger = tb_logger)
    trainer.fit(model)


if __name__=="__main__":

    main()
