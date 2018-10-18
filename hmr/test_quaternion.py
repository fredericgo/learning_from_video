from absl import flags
import src.config
import sys
from src.SkeletonExtractor import SkeletonExtractor
import numpy as np
import transforms3d
from transforms3d import quaternions, euler


config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1

e = SkeletonExtractor(config)
f = './data/coco1.png'
z, theta = e(f)
theta = theta.reshape((-1,3))
print(theta)
x = theta[1]
th = np.linalg.norm(x)
x[0], x[1], x[2] = x[2], x[0], x[1]
x /= th
print(x)
print(th)
q = quaternions.axangle2quat(x, th)
a = euler.quat2euler(q)
print(q)
print(a)

#np.save('results/k_tree.npy', data)
#np.save('j3d.npy', z)
