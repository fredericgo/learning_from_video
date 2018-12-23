
import os
import sys

root_path = os.path.realpath(__file__).split('/multipose_get_people.py')[0]
# os.chdir(root_path)
sys.path.append(root_path)

# from evaluate.tester import Tester
from evaluate.multi_pose_net_estimator import Params, MultiPoseNetEstimator


# Set Training parameters
params = Params()
params.backbone = 'resnet101'
params.subnet_name = 'both'
params.inp_size = 480  # input picture size = (inp_size, inp_size)
params.coeff = 2
params.in_thres = 0.21
params.ckpt = os.path.join(root_path, 'demo/models/ckpt_baseline_resnet101.h5')

two_d_estimator = MultiPoseNetEstimator(params)


def multipose_get_people(img):
    res = two_d_estimator.infer(img)
    return res
