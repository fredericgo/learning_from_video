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

    video_motion_process = VideoMotionProcessor(config)
    img_dir = "data/youtube/baseball_pitch/"
    video_motion_process(img_dir)




