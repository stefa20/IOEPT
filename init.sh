#!/bin/bash
set -e

## docker creation
docker build --network="host" -t humath/ioept:latest .

## run dettached container
docker run -d -it --rm \
    --name ioept  \
    --net="host" \
    -v "$PWD":/code \
    humath/ioept:latest
#    /bin/bash  #
  #--device /dev/video0 \
