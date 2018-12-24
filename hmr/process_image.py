from absl import flags
import os
import src.config
import sys
import numpy as np
import json
import tempfile
from src.video_processor import VideoMotionProcessor


if __name__ == "__main__":
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL

    skill = "b2"

    video_motion_process = VideoMotionProcessor(config)
    img_dir = "data/youtube/{}/".format(skill)
    motion_path = "/home/fredericgo/DeepMimic/data/motions/humanoid3d_{}.txt".format(skill)
    vis_path = "output"
    video_motion_process(img_dir, motion_path, vis_path)




