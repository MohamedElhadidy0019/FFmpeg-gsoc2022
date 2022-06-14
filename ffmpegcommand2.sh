./ffmpeg -v verbose \
-hwaccel cuda -hwaccel_output_format cuda -i input_green.mp4  \
-hwaccel cuda -hwaccel_output_format cuda -i static_blue.mp4 \
-init_hw_device cuda \
-filter_complex \
" \
[0:v]scale_cuda=format=yuv420p,dummy_cuda[overlay_video];
[1:v]scale_cuda=format=yuv420p[base];
[base][overlay_video]overlay_cuda" \
-an -sn -c:v h264_nvenc -cq 20 overlay_test.mp4

#scale_cuda=format=yuv420p,dummy_cuda