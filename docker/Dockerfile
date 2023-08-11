# syntax=docker/dockerfile:1

FROM ubuntu:22.04

COPY ./install_system_deps.sh /tmp/install_system_deps.sh
RUN chmod +x /tmp/install_system_deps.sh
RUN /bin/bash /tmp/install_system_deps.sh
RUN python3 --version

COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt
RUN python3 -m pip freeze
