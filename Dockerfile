FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    curl \
    sox \
    openssh-server \
    ffmpeg \
    libgl1-mesa-glx \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 安装 pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

COPY ./src/requirements.txt /app/requirements.txt
RUN pip install torch==2.6.0 torchaudio==2.6.0 \
    && pip install flash-attn==2.7.4.post1 --no-build-isolation \
    && pip install -r requirements.txt

# alias python3 as python
RUN ln -s /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]
