# IOEPT
Instance Of Emotion Perception Tool.

Emotion detection tool that works with images and videos, realtime inference is
provided though streamlit local web-based app and a FastAPI server is also provided
for custom code level integrations.

Prerequisites
-------------

- Docker

Configuration
-------------

Automatically create the docker image and run the detached services in the local host with:
```
chmod 777 init.sh  # give excecutable permission.
./init.sh
```

Realtime Video Inference can be visualized with our streamlit demo: 

```
http://0.0.0.0:82
```

Documentation and demo usage for the Image and Video API can be explored through: 

```
http://0.0.0.0:80/docs
```

### Install IOEPT Manualy from Dockerfile

First, build the docker image from the Dockerfile:

```
docker build --network="host" -t humath/ioept:latest .
```

Run the docker container:

```
docker run -d -it --rm \
    --name ioept  \
    --net="host" \
    --device /dev/video1 \
    -v "$PWD":/code \
    humath/ioept:latest 
```







