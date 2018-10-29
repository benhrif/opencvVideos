# USAGE
# python photo_booth.py --output output

# import the necessary packages
from __future__ import print_function
from pyimagesearch.darknetYoloGCP import PhotoBoothApp
from imutils.video import VideoStream
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# otherwise, grab a reference to the video file
# start the app
pba = PhotoBoothApp(args["tracker"])
#pba.root.mainloop()
