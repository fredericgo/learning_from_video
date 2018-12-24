ffmpeg -y -ss 00:01:48.3 -t 00:00:04 -i data/raw/pitch2.mp4 -r 50.0 -vf scale=640:480 data/frames/pitch2/img_%5d.jpg 

