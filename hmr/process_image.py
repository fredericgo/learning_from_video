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

	VideoMotionProcessor()


