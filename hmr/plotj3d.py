import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from transforms3d.axangles import axangle2mat

def p3d(joints):
    #(0 - r ankle,
    # 1 - r knee,
    # 2 - r hip,
    # 3 - l hip,
    # 4 - l knee,
    # 5 - l ankle,
    # 6 - pelvis,
    # 7 - thorax,
    # 8 - upper neck,
    # 9 - head top,
    # 10 - r wrist,
    # 10 - r wrist,
    # 12 - r shoulder,
    # 13 - l shoulder,
    # 14 - l elbow,
    # 15 - l wrist)
    limb_parents = [1, 2, 12, 12, 3, 4, 7, 8, 12, 12, 9, 10, 13, 13, 13]

    #y_temp = joints[:,1]
    R = axangle2mat([1,0,0], np.deg2rad(90))
    joints = joints.dot(R)
    print(joints.shape)
    #temp = joints[:,1]

    #joints[:,1] = joints[:,2]
    #joints[:,2] = -temp
    # plt.ion()
    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    #ax.view_init(elev=-70,azim=-90)
    #ax.scatter3D(joints[:,0],
    #          joints[:,1],
    #          joints[:,2], 'gray')
    idx = 13
    ax.scatter3D(joints[idx,0],
                 joints[idx,1],
                 joints[idx,2], c='red', s=100)

    for i in range(joints.shape[0]):
        x_pair = [joints[i, 0], joints[limb_parents[i], 0]]
        y_pair = [joints[i, 1], joints[limb_parents[i], 1]]
        z_pair = [joints[i, 2], joints[limb_parents[i], 2]]
        ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)

    #ax.axis('off')
    plt.show()
    #plt.savefig("test.jpg")

x = np.load('j3d.npy')
p3d(x)
