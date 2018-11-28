import os, sys
root_path = os.path.realpath(__file__).split('/multipose_test.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

from network.posenet import poseNet
from evaluate.tester import Tester
from evaluate.multi_pose_net_estimator import Params, MultiPoseNetEstimator


# Set Training parameters
params = Params()
params.backbone = 'resnet101'
params.subnet_name = 'both'
params.inp_size = 480  # input picture size = (inp_size, inp_size)
params.coeff = 2
params.in_thres = 0.21
params.testdata_dir = './demo/test_images/'
params.testresult_dir = './demo/output/'
params.testresult_write_image = True  # Whether to write result pictures
params.testresult_write_json = True# Whether to write json result
params.ckpt = './demo/models/ckpt_baseline_resnet101.h5'

two_d_estimator = MultiPoseNetEstimator(params)
img = './demo/test_images/pic1.jpg'
res = two_d_estimator.infer(img)
print(res)

