ffmpeg -y -ss 00:00:36 -t 00:00:40 -i raw/videoplayback.mp4 -r 50.0 -vf scale=640:480 frames/skating/skating_%5d.jpg 

