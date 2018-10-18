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

joints = {
    'L_Shoulder': 16, 'L_Elbow': 18,
    'R_Shoulder': 17, 'R_Elbow': 19,
    'L_Hip': 1,       'L_Knee': 4,
    'R_Hip': 2,       'R_Knee': 5
}


target_joints = {
    'L_Shoulder': [4,5,6],    'L_Elbow': [9],
    'R_Shoulder': [10,11,12], 'R_Elbow': [15],
    'L_Hip': [16,17,18],      'L_Knee': [21],
    'R_Hip': [23,24,25],      'R_Knee': [28]
}

def to_euler_xyz(x):
    x[0], x[1], x[2] = x[2], x[0], x[1]
    th = np.linalg.norm(x)
    x_norm = x / th
    q = quaternions.axangle2quat(x, th)
    a = euler.quat2euler(q)
    return a

def to_angle(x):
    th = np.linalg.norm(x)
    return th

z = np.zeros((30, 3))
for joi, num in joints.items():
    print("{}:".format(joi))
    x = theta[num]
    if joi in ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee']:
        a = to_angle(x)
    else:
        a = to_euler_xyz(x)
    z[target_joints[joi]] = a

np.save('k_tree.npy', z)
#np.save('j3d.npy', z)
