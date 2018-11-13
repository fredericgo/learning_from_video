from absl import flags
import os
import src.config
import sys
from src.MoRecSkeletonExtractor import MoRecSkeletonExtractor
import numpy as np
import json
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
#config.batch_size = 1

def p3d(joints, filename):
    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_zlim(0,10)
    ax.view_init(elev=-70,azim=-90)
    ax.scatter3D(joints[:,0],
              	 joints[:,1],
                 joints[:,2], 'gray')

    for i in range(joints3d.shape[0]):
        x_pair = [joints[i, 0], joints[limb_parents[i], 0]]
        y_pair = [joints[i, 1], joints[limb_parents[i], 1]]
        z_pair = [joints[i, 2], joints[limb_parents[i], 2]]
        ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
    ax.axis('off')
    plt.savefig(filename)

def plot_joints(joints3d):
    limb_parents = [1, 2, 12, 12, 3, 4, 7, 8, 12, 12, 9, 10, 13, 13, 13]

    joints3d = joints3d[:, :14, :]
    tempdir = tempfile.mkdtemp()
    print(tempdir)
    for i, x in enumerate(joints3d):
        outfilename = os.path.join(tempdir, 'temp_{:05d}.png'.format(i))
    	p3d(x, outfilename)


e = MoRecSkeletonExtractor(config)
img_dir = "data/youtube/baseball_pitch/"
z0, z_pred, j3d = e(img_dir, get_J3d=True)

mfile = dict()
mfile['Loop'] = 'wrap'
z_pred[:, 0] = 0.0625
z_pred[:, 4:8] = [1, 0, 0, 0]
mfile['Frames'] = z_pred.tolist()
 
with open('results/humanoid3d_pitch.txt', 'w') as f:
	json.dump(mfile, f, indent=2)


plot_joints(j3d)
