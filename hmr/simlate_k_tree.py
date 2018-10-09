import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer


model = load_model_from_path('humanoid.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

x = np.load('k_tree.npy')
x_rad = np.deg2rad(x)

sim_state = sim.get_state()
print(x)

while True:
    sim_state = sim.get_state()
    for i in range(11, 24):
        sim_state.qpos[i] = x_rad[i]
    sim.set_state(sim_state)
    sim.forward()
    #sim.step()
    viewer.render()
