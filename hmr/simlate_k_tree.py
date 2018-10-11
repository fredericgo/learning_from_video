import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import time

model = load_model_from_path('humanoid.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

x = np.load('results/k_trees.npy')

sim_state = sim.get_state()
#x =x[99]
step = 0
while True:
    y = x[step]
    sim_state = sim.get_state()
    #for i in range(3, 7):
    #    sim_state.qpos[i] = y[i]
    for i in range(11, 24):
        sim_state.qpos[i] = np.deg2rad(y[i])
    sim.set_state(sim_state)
    sim.forward()
    #sim.step()
    viewer.render()
    step += 1
    time.sleep(0.1)
