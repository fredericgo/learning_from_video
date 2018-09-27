import gym

env = gym.make('Humanoid-v2')
x = env.reset()
print(x.shape)
