echo kak
./ffmpeg -v verbose \
-hwaccel cuda -hwaccel_output_format cuda -i input_green.mp4  \
-init_hw_device cuda \
-filter_complex \
" \
[0:v]dummy_cuda[overlay_video];
[overlay_video]format=yuv420p" \
-an -sn -c:v h264_nvenc -cq 20 overlay_test2.mp4

mpv overlay_test2.mp4
#scale_cuda=format=yuv420p,dummy_cuda
