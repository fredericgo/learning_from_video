ffmpeg -ss 00:00:00 -t 00:00:03 -i data/raw/v_BaseballPitch_g01_c01.avi -r 25.0 -vf scale=640:480 data/frames/baseball/baseball_%5d.jpg 

