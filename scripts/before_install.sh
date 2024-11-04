#!/bin/bash
set -e

echo "Updating package list..."
apt-get update

echo "Installing necessary packages..."
apt-get install -y python3-full python3-pip python3-venv

echo "Creating /home/ubuntu/app directory if it doesn't exist..."
mkdir -p /home/ubuntu/app

echo "Changing ownership of /home/ubuntu/app to ubuntu:ubuntu..."
chown -R ubuntu:ubuntu /home/ubuntu/app

echo "Ownership and permissions of /home/ubuntu/app after chown:"
ls -ld /home/ubuntu/app
