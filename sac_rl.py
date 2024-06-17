#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
import torch.optim as optim




# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu


environment_dim = 20




class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(SACActor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_mean = nn.Linear(hidden_size, action_dim)
        self.fc3_log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        log_std = self.fc3_log_std(x)
        return mean, log_std

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(SACCritic, self).__init__()

        # First Critic
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_q1 = nn.Linear(hidden_size, hidden_size)
        self.fc3_q1 = nn.Linear(hidden_size, 1)

        # Second Critic
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_q2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_q2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # First Critic
        x_q1 = torch.cat([state, action], dim=1)
        x_q1 = F.relu(self.fc1_q1(x_q1))
        x_q1 = F.relu(self.fc2_q1(x_q1))
        q1 = self.fc3_q1(x_q1)

        # Second Critic
        x_q2 = torch.cat([state, action], dim=1)
        x_q2 = F.relu(self.fc1_q2(x_q2))
        x_q2 = F.relu(self.fc2_q2(x_q2))
        q2 = self.fc3_q2(x_q2)

        return q1, q2


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        # Initialize the Actor network
        self.actor = SACActor(state_dim, action_dim, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = SACCritic(state_dim, action_dim, hidden_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        # Temperature parameter for entropy regularization
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.max_action = max_action
        self.writer = SummaryWriter(log_dir="./sac_ros/src/td3/scripts/runs")
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        action = normal.sample()
        action = torch.tanh(action).detach().numpy().flatten()
        return action

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=64,
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = float('-inf')
        av_loss = 0

        for it in range(iterations):
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            current_action, log_std = self.actor(state)
            current_std = torch.exp(log_std)
            current_normal = Normal(current_action, current_std)
            current_log_prob = current_normal.log_prob(action).sum(axis=-1, keepdim=True)

            current_Q1, current_Q2 = self.critic(state, action)

            actor_loss = (self.alpha * current_log_prob - torch.min(current_Q1, current_Q2)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            next_action, next_log_std = self.actor(next_state)
            next_std = torch.exp(next_log_std)
            next_normal = Normal(next_action, next_std)
            next_log_prob = next_normal.log_prob(next_action).sum(axis=-1, keepdim=True)
            target_Q1, target_Q2 = self.critic(state, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))

            target_Q = reward + ((1 - done) * discount * (target_Q - self.alpha * next_log_prob)).detach()

            current_Q1, current_Q2 = self.critic(state, action)

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            av_loss += critic_loss.item()

            # Update temperature parameter (alpha)
            alpha_loss = -(self.log_alpha * (current_log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            if it % policy_freq == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter_count += 1
        env.get_logger().info(f"writing new results for a tensorboard")
        env.get_logger().info(f"loss, Av.Q, Max.Q, iterations: {av_loss / iterations}, {av_Q / iterations}, {max_Q}, {self.iter_count}")
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))



