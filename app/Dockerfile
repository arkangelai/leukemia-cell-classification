FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
RUN apt-get update && apt-get install -y git python3-dev python3-pip gcc libglib2.0-dev libsm6 libxext6 libxrender-dev libgl1-mesa-glx unzip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
ADD . / ./
ENV PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD ["gunicorn", "app:app", "-b", ":8080", "--timeout", "300"]
