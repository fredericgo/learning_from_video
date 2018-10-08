
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from ikpy import plot_utils
import matplotlib.pyplot as plt


from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np


def solve_one_angle(v_o, v_new, a1, b1):
    chain = Chain(name='arm', links=[
        #OriginLink(),
        URDFLink(
          name="shoulder1",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          bounds=b1,
          rotation=a1,
        ),
         URDFLink(
          name="elbow",
          translation_vector=v_o,
          orientation=[0, 0, 0],
          rotation=[0, 0, 0],
        )
    ])
    target_frame = np.eye(4)
    target_frame[:3, 3] = v_new
    angles = chain.inverse_kinematics(target_frame)[:2]
    return np.rad2deg(angles)

def solve_shoulder_angles(v_o, v_new, a1, a2, b1, b2):
    chain = Chain(name='arm', links=[
        #OriginLink(),
        URDFLink(
          name="shoulder1",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          bounds=b1,
          rotation=a1,
        ),
        URDFLink(
          name="shoulder2",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          bounds=b2,
          rotation=a2,
        ),
         URDFLink(
          name="elbow",
          translation_vector=v_o,
          orientation=[0, 0, 0],
          rotation=[0, 0, 0],
        )
    ])
    target_frame = np.eye(4)
    target_frame[:3, 3] = v_new
    angles = chain.inverse_kinematics(target_frame)[:2]
    return np.rad2deg(angles)


def solve_hip_angles(v_o, v_new, a1, a2, a3, b1, b2, b3):
    chain = Chain(name='arm', links=[
        #OriginLink(),
        URDFLink(
          name="hip1",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          bounds=b1,
          rotation=a1,
        ),
        URDFLink(
          name="hip2",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          bounds=b2,
          rotation=a2,
        ),
        URDFLink(
          name="hip3",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          bounds=b3,
          rotation=a3,
        ),
         URDFLink(
          name="knee",
          translation_vector=v_o,
          orientation=[0, 0, 0],
          rotation=[0, 0, 0],
        )
    ])
    target_frame = np.eye(4)
    target_frame[:3, 3] = v_new
    angles = chain.inverse_kinematics(target_frame)[:3]
    return np.rad2deg(angles)

def solve_l_shoulder_angles(v_old, v_new):
    a1 = np.array([2,-1, 1])
    a2 = np.array([0, 1, 1])
    b1 = (-68, 85)
    b2 = (-68, 85)
    return solve_shoulder_angles(v_old, v_new, a1, a2, b1, b2)

def solve_r_shoulder_angles(v_old, v_new):
    a1 = np.array([2, 1, 1])
    a2 = np.array([0, -1, 1])
    b1 = (-85, 60)
    b2 = (-85, 60)
    return solve_shoulder_angles(v_old, v_new, a1, a2, b1, b2)

def solve_l_hip_angles(v_old, v_new):
    a1 = np.array([-1, 0, 0])
    a2 = np.array([0, 0, -1])
    a3 = np.array([0, 1, 0])
    b1 = (-25, 5)
    b2 = (-60, 35)
    b3 = (-120, 20)
    return solve_hip_angles(v_old, v_new, a1, a2, a3, b1, b2, b3)

def solve_r_hip_angles(v_old, v_new):
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 0, 1])
    a3 = np.array([0, 1, 0])
    b1 = (-25, 5)
    b2 = (-60, 35)
    b3 = (-120, 20)
    return solve_hip_angles(v_old, v_new, a1, a2, a3, b1, b2, b3)

def solve_r_elbow_angles(v_old, v_new):
    a1 = np.array([0, -1, 1])
    b1 = (-90, 50)
    return solve_one_angle(v_old, v_new, a1, b1)

if __name__ == "__main__":
    v_o = np.array([1,2,3])
    v_new = np.array([2,1,2])
    a1 = np.array([0, 0, 1])
    a2 = np.array([1, 0, 0])
    a3 = np.array([0, 1, 0])
    ans = solve_hip_angles(v_o, v_new, a1, a2, a3)
    print(ans)
