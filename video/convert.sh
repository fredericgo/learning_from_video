ffmpeg -y -ss 00:00:05.9  -t 00:00:03 -i data/raw/broad_jump.mp4 -r 50.0 -vf scale=1280:720 data/frames/broad_jump/img_%5d.jpg 

