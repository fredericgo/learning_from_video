
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from ikpy import plot_utils
import matplotlib.pyplot as plt


from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np

def solve_shoulder_angles(v_o, v_new, a1, a2):
    chain = Chain(name='arm', links=[
        #OriginLink(),
        URDFLink(
          name="shoulder1",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          rotation=a1,
        ),
        URDFLink(
          name="shoulder2",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
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


def solve_hip_angles(v_o, v_new, a1, a2, a3):
    chain = Chain(name='arm', links=[
        #OriginLink(),
        URDFLink(
          name="hip1",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          rotation=a1,
        ),
        URDFLink(
          name="hip2",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
          rotation=a2,
        ),
        URDFLink(
          name="hip3",
          translation_vector=[0, 0, 0],
          orientation=[0, 0, 0],
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


if __name__ == "__main__":
    v_o = np.array([1,2,3])
    v_new = np.array([2,1,2])
    a1 = np.array([0, 0, 1])
    a2 = np.array([1, 0, 0])
    a3 = np.array([0, 1, 0])
    ans = solve_hip_angles(v_o, v_new, a1, a2, a3)
    print(ans)
