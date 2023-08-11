#!/usr/bin/bash

set -eux o pipefail

dependencies=(python3.10 python3-pip nano)

apt-get update
apt-get install -y ${dependencies[*]}
