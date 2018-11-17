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
f = './data/coco5.png'
z, theta = e(f)
theta = theta.reshape((-1,3))

joints = {
    'Spine1': 3,
    'L_Shoulder': 16, 'L_Elbow': 18,
    'R_Shoulder': 17, 'R_Elbow': 19,
    'L_Hip': 1,       'L_Knee': 4,
    'R_Hip': 2,       'R_Knee': 5
}

target_joints = {
    'Spine1': [0,1,2],
    'L_Shoulder': [6,7,8],    'L_Elbow': [11],
    'R_Shoulder': [12,13,14], 'R_Elbow': [17],
    'L_Hip': [18,19,20],      'L_Knee': [23],
    'R_Hip': [25,26,27],      'R_Knee': [30]
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

z = np.zeros(32)
for joi, num in joints.items():
    print("{}:".format(joi))
    x = theta[num]
    if joi in ['L_Elbow']:
        a = -to_angle(x)
    elif joi in ['R_Elbow', 'R_Knee', 'L_Knee']:
        a = to_angle(x)
    else:
        a = to_euler_xyz(x)
    print(a)
    z[target_joints[joi]] = a

np.save('k_tree.npy', z)
#np.save('j3d.npy', z)
