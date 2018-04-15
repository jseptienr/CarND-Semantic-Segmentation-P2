#!/bin/bash
# this script is meant to be used to download the VGG model and data set
# specially when using a new aws instance for GPU access
# based on Udacity forums post credit to @driveWell. thanks!
cd data
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip
unzip vgg.zip
rm vgg.zip
wget http://kitti.is.true.mpg.de/kitti/data_road.zip
unzip data_road.zip
rm data_road.zip
