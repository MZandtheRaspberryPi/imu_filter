#!/usr/bin/bash

dependencies=(python3-pip nano)

sudo apt-get update
sudo apt-get install -y ${dependencies[*]}