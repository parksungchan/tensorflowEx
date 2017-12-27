Installing OpenCV 3.2.0 with contrib modules in Ubuntu 16.04 LTS
https://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/
================================================================
cd /home/dev

sudo apt-get update

sudo apt-get upgradesudo

apt-get dist-upgrade

sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff5-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev default-jdk ant libvtk5-qt4-dev

mkdir opencv

cd opencv

wget https://github.com/opencv/opencv/archive/3.2.0.tar.gz

tar -xvzf 3.2.0.tar.gz

wget https://github.com/opencv/opencv_contrib/archive/3.2.0.zip

unzip 3.2.0.zip

cd opencv-3.2.0

sed - i
's/${freetype2_LIBRARIES} ${harfbuzz_LIBRARIES}/${FREETYPE_LIBRARIES} ${HARFBUZZ_LIBRARIES}/g'.. / opencv_contrib - 3.2
.0 / modules / freetype / CMakeLists.txt
########################################################################################################################
mkdir build

cd build

cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON .. -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.2.0/modules ..

make

sudo make install

echo '/usr/local/lib' | sudo tee --append /etc/ld.so.conf.d/opencv.conf

sudo ldconfig

echo 'PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig' | sudo tee --append ~/.bashrc

echo 'export PKG_CONFIG_PATH' | sudo tee --append ~/.bashrc

source ~/.bashrc
########################################################################################################################


########################################################################################################################
Example:
mkdir saliency
cd saliency
cp ../opencv/opencv_contrib-3.2.0/modules/saliency/samples/computeSaliency.cpp .
cp ../opencv/opencv-3.2.0/samples/data/Megamind.avi .
g++ -o computeSaliency `pkg-config opencv --cflags` computeSaliency.cpp `pkg-config opencv --libs`
./computeSaliency FINE_GRAINED Megamind.avi 23


sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake git pkg-config
sudo apt autoremove
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo pip3 install virtualenv virtualenvwrapper
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
source ~/.bashrc
mkvirtualenv cv    <-- library 추가 설치를 할 수 있다.
sudo apt-get install python3.5-dev
pip install numpy

git clone https://github.com/Itseez/opencv.git
cd /home/dev/opencv
git checkout 3.0.0

git clone https://github.com/Itseez/opencv_contrib.git
cd /home/dev/opencv_contrib
git checkout 3.0.0

cd /home/dev/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..

make -j4
sudo make install
sudo ldconfig

ls -l /usr/local/lib/python3.5/site-packages/

cd ~/.virtualenvs/cv/lib/python3.5/site-packages/
ln -s /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so cv2.so

pip install tensorflow
pip install matplotlib
pip install wget
pip install scipy
pip install imutils
pip install sklearn
pip install requests
########################################################################################################################
workon cv
python
>>> import cv2
>>> cv2.__version__
'3.0.0'
########################################################################################################################
Last Your Python Interpreter Change
/root/.virtualenvs/cv/lib/python3.5/site-packages




참조 : https://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/














