# [Note, Feb 23, 2022]: Containers using the following image have problems accessing GPUs.
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel 
# [Note, Feb 23, 2022]: Containers using the following image can access GPUs.
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN mkdir GraphBERT
WORKDIR /GraphBERT

# RUN apt update
# RUN apt install -y tmux nano htop

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt