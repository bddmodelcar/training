FROM floydhub/pytorch:latest-gpu-py2
MAINTAINER Tushar Pankaj (tushar.s.pankaj@gmail.com)
RUN apt-get install libopencv-dev python-opencv
RUN pip install termcolor pyserial pandas
