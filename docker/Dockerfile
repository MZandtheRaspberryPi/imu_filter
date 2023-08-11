# syntax=docker/dockerfile:1

FROM ubuntu:22.04

COPY ./docker/install_system_deps.sh /tmp/install_system_deps.sh
RUN chmod +x /tmp/install_system_deps.sh

RUN /bin/bash /tmp/install_system_deps.sh

COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt