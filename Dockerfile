FROM coffeateam/coffea-base:latest

# Build the image as root user
USER root

RUN pip3 install keras tensorflow sklearn yahist

COPY . /tW_scattering
WORKDIR /tW_scattering

RUN echo "export PYTHONPATH=${PYTHONPATH}:/tW_scattering" >> ~/.bashrc

# Run as docker user by default when the container starts up
#USER docker
