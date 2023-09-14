# IOPT Docker Container
FROM ubuntu:20.04

## instal dependences 
RUN apt-get update && apt-get install -y software-properties-common 
RUN apt-get update 
RUN apt-get install -y ffmpeg libsm6 libxext6 

# install python 3.8
#RUN apt-get install -y python3-pip
RUN apt-get install -y python3.8 curl
RUN apt-get install -y python3.8-distutils

# install and upgrade pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN python3 -m pip install -U pip

#install python packages
RUN pip install fastapi "uvicorn[standard]" pandas python-multipart
RUN pip install torch facenet-pytorch opencv-python
RUN pip install streamlit plotly

WORKDIR /code
ADD serve.sh /code/

EXPOSE 80
EXPOSE 82
RUN chmod +x serve.sh
ENTRYPOINT ["sh", "serve.sh"]