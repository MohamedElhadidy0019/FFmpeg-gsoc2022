./ffmpeg -y \
-hwaccel cuda -hwaccel_output_format cuda -i input_green.mp4 -vf \
-hwaccel cuda -hwaccel_output_format cuda -i static_blue.mp4 -vf \
-filter_complex \

" \
[0:v]dummy_cuda[base];
[1:v]scale_npp=640:-2:format=nv12[overlay_video];
[base][overlay_video]overlay_cuda" \
-an -c:v h264_nvenc overlay_test.mp4


#-an -c:v h264_nvenc overlay_test.mp4



# " \
# [0:v]dummy_cuda[base];
# [1:v]format=nv12[overlay_video];
# [base][overlay_video]overlay_cuda=x=640:y=0" \

./ffmpeg -v verbose -hwaccel cuda -hwaccel_output_format cuda -init_hw_device cuda -i input_green.mp4 -vf dummy_cuda -an -sn -c:v h264_nvenc -cq 20 -t 00:00:30 -y output.mp4