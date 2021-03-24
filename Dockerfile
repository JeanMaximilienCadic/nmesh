FROM python:3.6

RUN apt update -y && apt-get install libgl1-mesa-dev xvfb -y

WORKDIR /workspace

RUN pip install git+https://github.com/JeanMaximilienCadic/gnutools-python
RUN pip install git+https://github.com/JeanMaximilienCadic/nmesh
