FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL authors="eduardo"

RUN apt-get -qq update && \
    apt-get -qq install -y build-essential python3-pip && \
    pip3 install pycuda flask numpy

COPY . /app
WORKDIR /app
EXPOSE 5000

CMD ["python3", "./app.py"]
