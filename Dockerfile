FROM python:3.6

RUN apt update -y && apt-get install libgl1-mesa-dev xvfb -y

COPY . /workspace/

WORKDIR /workspace

RUN pip install git+https://github.com/JeanMaximilienCadic/nmesh