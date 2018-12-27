VIDEONAME=mandance.mp4
TRICKNAME=rotation

mkdir data/frames/$TRICKNAME

ffmpeg -y -ss 00:00:21 -t 00:00:02 -i data/raw/$VIDEONAME -r 50.0 -vf scale=1280:720 data/frames/$TRICKNAME/img_%5d.jpg 

