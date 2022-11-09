FROM python:3.8

RUN pip install --upgrade pip

RUN mkdir /tf
RUN mkdir -p /tf/video
RUN mkdir -p /tf/models
COPY requirements.txt /tf/requirements.txt
WORKDIR /tf

RUN git clone https://github.com/tensorflow/models /tf/models

ENV PROTOC_ZIP=protoc-3.13.0-linux-x86_64.zip
RUN apt-get update && apt-get install -y unzip
RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/$PROTOC_ZIP \
    && unzip -o $PROTOC_ZIP -d /usr/local bin/protoc \
    && unzip -o $PROTOC_ZIP -d /usr/local 'include/*' \ 
    && rm -f $PROTOC_ZIP

WORKDIR /tf/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .  
RUN python -m pip install . 

WORKDIR /tf

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt
COPY . /tf
CMD ["python", "/tf/OD_v3.py"]