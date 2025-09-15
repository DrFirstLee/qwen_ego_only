#!/bin/bash
set -e

# tmux 자동 실행 방지
touch ~/.no_auto_tmux

# 기본 패키지 업데이트 및 설치
sudo apt update -y
sudo apt install -y libgl1-mesa-glx nano zip wget

# Anaconda 자동 설치
ANACONDA_SH=Anaconda3-2024.02-1-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/$ANACONDA_SH
bash $ANACONDA_SH -b -p $HOME/anaconda3

# PATH 설정
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Conda 버전 확인
conda --version

# gdown 설치 및 데이터 다운로드
pip install --upgrade pip
pip install gdown
gdown 1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk

# 데이터 압축 해제
unzip AGD20K.zip
