Step 1: Install prerequisites

Upgrade any pre-installed packages:

Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ sudo apt-get update
$ sudo apt-get upgrade

Install developer tools used to compile OpenCV 3.0:
Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ sudo apt-get install build-essential cmake git pkg-config

Install libraries and packages used to read various image formats from disk:
Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev

Install a few libraries used to read video formats from disk:
Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

Install GTK so we can use OpenCV’s GUI features:
Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ sudo apt-get install libgtk2.0-dev

Install packages that are used to optimize various functions inside OpenCV, such as matrix operations:
Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ sudo apt-get install libatlas-base-dev gfortran



Step 2: Setup Python (Part 2)

We’re halfway done setting up Python. But in order to compile OpenCV 3.0 with Python 3.4+ bindings, we’ll need to install the Python 3.4+ headers and development files:
Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ sudo apt-get install python3.4-dev

OpenCV represents images as NumPy arrays, so we need to install NumPy into our cv  virtual environment:
Install OpenCV 3.0 and Python 3.4+ on UbuntuShell
$ pip install numpy

If you end up getting a Permission denied error related to pip’s .cache  directory, like this:
nstall OpenCV 3.0 and Python 3.4+ on UbuntuShell

$ sudo rm -rf ~/.cache/pip/
$ pip install numpy



Step 3: Build and install OpenCV 3.0 with Python 3.4+ bindings
$ cd ~
$ git clone https://github.com/Itseez/opencv.git
$ cd opencv
$ git
checkout
3.0
.0
























































