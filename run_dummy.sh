#!/bin/bash

make -j 20
./ffmpeg -v verbose -hwaccel cuda -hwaccel_output_format cuda -init_hw_device cuda -i $1 -vf  scale_cuda=format=yuv420p,dummy_cuda -an -sn -c:v h264_nvenc -cq 20 -t 00:00:30 -y out.mp4
#scale_cuda=format=yuv420p
mpv out.mp4
#rm out.mp4







