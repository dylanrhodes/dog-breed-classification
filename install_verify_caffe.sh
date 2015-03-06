# Download and run docker container
DOCKER_NVIDIA_DEVICES="--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm"
sudo docker run -v /tmp:/root -ti $DOCKER_NVIDIA_DEVICES tleyden5iwx/caffe-gpu-master /bin/bash

# Train MNIST LeNet to verify caffe installation
#cd /opt/caffe/data/mnist
#./get_mnist.sh
#cd /opt/caffe
#./examples/mnist/create_mnist.sh
#./examples/mnist/train_lenet.sh
