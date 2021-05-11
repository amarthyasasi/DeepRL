import matplotlib.pyplot as plt
import os,sys
import numpy as np

fo = open("./epoch_reward.txt", "r+")
reward_strs = fo.readlines()
print(len(reward_strs))
rewards = []

for i in range(len(reward_strs)):
    rewards.append(np.float32(reward_strs[i][:2]))

plt.plot(rewards,c = 'orange')
plt.grid(True)
plt.title("DQN: Rewards vs Epochs")
plt.savefig("rewards.png",dpi = 300)

fo2 = open("./epoch_loss.txt", "r+")
loss_strs = fo2.readlines()
print(len(loss_strs))
losses = []

for i in range(len(reward_strs)):
    strs = loss_strs[i].split(",")[0].split("(")[1]
    losses.append(np.float32(strs))

plt.plot(losses,c = 'orange')
plt.ylim(0,3000)
plt.grid(True)
plt.title("DQN: Losses vs Epochs")
plt.savefig("losses.png",dpi = 300)