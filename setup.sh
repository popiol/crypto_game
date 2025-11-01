#!/bin/bash

# install python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo rm /usr/bin/python3
sudo ln -s python3.11 /usr/bin/python3
sudo ln -s python3.11 /usr/bin/python
sudo apt install python3.11-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# install python requirements
python -m pip install -r requirements.txt

# install aws cli
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
