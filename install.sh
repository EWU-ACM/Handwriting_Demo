sudo apt-get update;
sudo apt-get install -y build-essential gcc g++ curl \
                cmake libreadline-dev git-core libqt4-core libqt4-gui \
                libqt4-dev libjpeg-dev libpng-dev ncurses-dev \
                imagemagick libzmq3-dev gfortran unzip gnuplot \
                gnuplot-x11 ipython;

git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh

