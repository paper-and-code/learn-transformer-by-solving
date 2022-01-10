FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install build-essential --yes

RUN apt-get install git --yes
RUN apt-get install libopencv-dev --yes 
COPY ./DETR/requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install opencv-python
