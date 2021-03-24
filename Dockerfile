FROM jcadic/gnutools-python

RUN apt update -y && apt-get install libgl1-mesa-dev xvfb -y

WORKDIR /workspace

RUN pip install git+https://github.com/JeanMaximilienCadic/nmesh
