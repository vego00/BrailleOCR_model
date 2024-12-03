#!/bin/bash
set -e

echo "Updating package list..."
apt-get update

echo "Installing necessary packages..."
apt-get install -y software-properties-common

add-apt-repository ppa:deadsnakes/ppa
apt-get update

apt-get install -y python3.8 python3.8-venv python3.8-dev python3-pip

# 'python3' 심볼릭 링크 변경을 제거합니다.
# echo "Setting python3 to point to python3.8..."
# update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

echo "Creating /home/ubuntu/app directory if it doesn't exist..."
mkdir -p /home/ubuntu/app

echo "Changing ownership of /home/ubuntu/app to ubuntu:ubuntu..."
chown -R ubuntu:ubuntu /home/ubuntu/app

echo "Ownership and permissions of /home/ubuntu/app after chown:"
ls -ld /home/ubuntu/app