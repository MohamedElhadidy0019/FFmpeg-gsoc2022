#!/bin/bash
make -j 10
x=1
while [ $x -le 2 ]
do
    rm overlay_test.mp4
    ./ffmpeg -v verbose \
    -hwaccel cuda -hwaccel_output_format cuda -i input_green.mp4  \
    -hwaccel cuda -hwaccel_output_format cuda -i static_blue.mp4 \
    -init_hw_device cuda \
    -filter_complex \
    " \
    [0:v]scale_cuda=format=yuv420p,dummy_cuda=0x25FF13:0.1:0.12[overlay_video];
    [1:v]scale_cuda=format=yuv420p[base];
    [base][overlay_video]overlay_cuda" \
    -an -sn -c:v h264_nvenc -cq 20 overlay_test.mp4
    x=$(( $x + 1 ))
done
mpv overlay_test.mp4
rm overlay_test.mp4
#scale_cuda=format=yuv420p,dummy_cuda