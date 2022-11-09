This project tries to solve the code challenge which is saved in this repo too.

Please let the video on the docker_v1/video folder and the execute the Dockerfile:

docker build -t docker_perf_v3 .

Then, once the contailer is built, you could run the docker container allowing the interactive permissiones as well as specifying the local route to upload and video and recieve the processed videa and the summary.txt file too.

docker run -i -t -v "local_route_where_the_original_video_is_located":/tf/video docker_perf_v3