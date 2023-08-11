#! /bin/bash

set -eux o pipefail

cp requirements.txt ./docker/requirements.txt
cd docker
docker build -f Dockerfile --progress=plain .
rm requirements.txt
cd ..
