opencv Install
1. sudo apt-get update

2. 컴파일시 필요한 패키지들 설치
$ sudo apt-get install build-essential checkinstall cmake git pkg-config yasm libtiff5-dev libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev  libv4l-dev python-dev python-numpy libtbb-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils libopenexr-dev python-tk  libeigen3-dev libx264-dev

$ sudo add-apt-repository ppa:mc3man/gstffmpeg-keep
$ sudo apt-get update
$ sudo apt-get install ffmpeg gstreamer0.10-ffmpeg

3. OpenCV 소스코드 다운로드
$ mkdir opencv_source
$ cd opencv_source
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git

4. OpenCV 설정
$ cd opencv
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr \
-D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON \
-D WITH_TBB=ON -D WITH_IPP=OFF \
-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules  ../

5. OpenCV 컴파일 후 설치
$ make  -j $(nproc)
$ sudo make install
$ sudo ldconfig

6. opencv  버전 확인 및 예제 컴파일
$ pkg-config --modversion opencv
3.1.0






1. g++, cmake를 설치한다.
sudo apt-get install g++
sudo apt-get install cmake

http://opencv.org/
https://github.com/opencv/opencv/releases/tag/3.2.0


pkg-config --modversion opencv
########################################################################################################################
1. 만약 CUDA또는 opencv가 설치되어 있다면 사용하기 위해서 vi에디터 등으로 Makefile을 연다.
vi Makefile

CUDA를 사용한다면 GPU=1로

opencv를 사용한다면 OPENCV=1로 변경해준다.

(여기서는 opencv만 사용하였다.)


2. 다운받은 폴더로 이동하여 make한다.
cd darknet
make

./darknet imtest data/eagle.jpg


./darknet imtest data/horses.jpg

########################################################################################################################
wget http://pjreddie.com/media/files/yolo.weights

./darknet yolo test cfg/yolo.cfg yolo.weights data/dog.jpg















