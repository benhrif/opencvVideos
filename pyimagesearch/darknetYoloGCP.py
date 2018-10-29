# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from PIL import Image as ImagePIL
from PIL import ImageTk
import tkinter as tki
from tkinter import filedialog
import threading
import datetime
import imutils
import cv2
import os
import time
import numpy as np
import sys
import os.path
import random
import csv
from pydarknet import Detector, Image

class PhotoBoothApp:
	def __init__(self,trackerType):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.videoPath="videos/video2.avi"
		self.isPlay=True
		self.isPause=False
		self.isRestart=True
		self.time = 0
		self.frameCounting = 0
		self.fps = 1

		
		# Initialize YOLO Detection parameters
		self.confThreshold = 0.1  #Confidence threshold
		self.nmsThreshold = 0.4   #Non-maximum suppression threshold
		self.inpWidth = 320       #Width of network's input image
		self.inpHeight = 320     #Height of network's input image
		
		self.sec = 0
		self.frameRate = 0.1

		self.net = Detector(bytes("yolov3.cfg", encoding="utf-8"), bytes("yolov3.weights", encoding="utf-8"), 0, bytes("coco.data",encoding="utf-8"))
		

		# initialize a dictionary that maps strings to their corresponding
		# OpenCV object tracker implementations
		self.OPENCV_OBJECT_TRACKERS = {
			"csrt": cv2.TrackerCSRT_create,
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create,
			"mosse": cv2.TrackerMOSSE_create
		}
		self.e1 = (672, 191, 5, 58)
		self.e2 = (415, 243, 10, 134)
		self.e3 = (443, 49, 73, 8)
		self.e4 = (0, 0, 0, 0)
		self.e5 = (0, 0, 0, 0)
		
		self.s1 = (714, 252, 14, 62)
		self.s2 = (335, 115, 18, 110)
		self.s3 = (572, 120, 78, 17)
		self.s4 = (0, 0, 0, 0)
		self.s5 = (0, 0, 0, 0)
		self.frame = None
		self.crop_frame = None
		self.thread = None
		self.stopEvent = None
		
		# initialize OpenCV's special multi-object tracker
		self.trackers = cv2.MultiTracker_create()         
		self.trackerType = trackerType 

		self.trackingMatrix = [[]]
		self.countTracker = 0

	
		

		# start a thread that constantly pools the video sensor for
		# the most recently read frame	
				
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()


		with open('eggs.csv', 'w', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',	quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow(['Type', 'Entree', 'Sortie', 'Time'])



	def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				if self.isRestart == True:
						self.vs = cv2.VideoCapture(self.videoPath)
						self.isRestart = False
				if self.isPlay:
					# grab the frame from the video stream and resize it to
					# have a maximum width of 800 pixels
					#self.sec=self.sec+self.frameRate
					#self.vs.set(cv2.CAP_PROP_POS_MSEC,self.sec*1000)
					self.frame = self.vs.read()
					self.frame = self.frame[1]
					self.frameCounting = self.frameCounting + 1
					#fps = 30
					self.frame = imutils.resize(self.frame, width=800)
					(success, boxes) = self.trackers.update(self.frame)
					self.time = self.frameCounting / self.fps
					i=0;
					for box in boxes:
						(x, y, w, h) = [int(v) for v in box]
						if (self.crossS1((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s1")
								self.trackingMatrix[i].append(self.time)
								print(self.trackingMatrix)
						if (self.crossS2((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s2")
								self.trackingMatrix[i].append(self.time)
								print(self.trackingMatrix)
						if (self.crossS3((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s3")
								self.trackingMatrix[i].append(self.time)
								print(self.trackingMatrix)
						if (self.crossS4((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s4")
								self.trackingMatrix[i].append(self.time)
								print(self.trackingMatrix)
						if (self.crossS5((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s5")
								self.trackingMatrix[i].append(self.time)
								print(self.trackingMatrix)
						i=i+1
						
					
					if self.e1[3] > 0 :
						blob = Image(self.frame)
						results = self.net.detect(blob)

						self.postprocess(results)
				
		except RuntimeError as e:
			print("[INFO] caught a RuntimeError")
        
	def crossE1(self,rect):
		if((self.e1[0]<=(rect[2]+rect[0])/2 and self.e1[0] + self.e1[2] >= (rect[2]+rect[0])/2) and (self.e1[1]<=(rect[3]+rect[1])/2 and self.e1[1] + self.e1[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e1")
			return True;
		else :
			return False;
	def crossS1(self,rect):
		if((self.s1[0]<=(rect[2]+rect[0])/2 and self.s1[0] + self.s1[2] >= (rect[2]+rect[0])/2) and (self.s1[1]<=(rect[3]+rect[1])/2 and self.s1[1] + self.s1[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s1")
			return True;
		else :
			return False;	
	def crossE2(self,rect):
		if((self.e2[0]<=(rect[2]+rect[0])/2 and self.e2[0] + self.e2[2] >= (rect[2]+rect[0])/2) and (self.e2[1]<=(rect[3]+rect[1])/2 and self.e2[1] + self.e2[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e2")
			return True;
		else :
			return False;
	def crossS2(self,rect):
		if((self.s2[0]<=(rect[2]+rect[0])/2 and self.s2[0] + self.s2[2] >= (rect[2]+rect[0])/2) and (self.s2[1]<=(rect[3]+rect[1])/2 and self.s2[1] + self.s2[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s2")
			return True;
		else :
			return False;	

	def crossE3(self,rect):
		if((self.e3[0]<=(rect[2]+rect[0])/2 and self.e3[0] + self.e3[2] >= (rect[2]+rect[0])/2) and (self.e3[1]<=(rect[3]+rect[1])/2 and self.e3[1] + self.e3[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e3")
			return True;
		else :
			return False;
	def crossS3(self,rect):
		if((self.s3[0]<=(rect[2]+rect[0])/2 and self.s3[0] + self.s3[2] >= (rect[2]+rect[0])/2) and (self.s3[1]<=(rect[3]+rect[1])/2 and self.s3[1] + self.s3[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s3")
			return True;
		else :
			return False;


	def crossE4(self,rect):
		if((self.e4[0]<=(rect[2]+rect[0])/2 and self.e4[0] + self.e4[2] >= (rect[2]+rect[0])/2) and (self.e4[1]<=(rect[3]+rect[1])/2 and self.e4[1] + self.e4[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e4")
			return True;
		else :
			return False;
	def crossS4(self,rect):
		if((self.s4[0]<=(rect[2]+rect[0])/2 and self.s4[0] + self.s4[2] >= (rect[2]+rect[0])/2) and (self.s4[1]<=(rect[3]+rect[1])/2 and self.s4[1] + self.s4[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s4")
			return True;
		else :
			return False;	

	def crossE5(self,rect):
		if((self.e5[0]<=(rect[2]+rect[0])/2 and self.e5[0] + self.e5[2] >= (rect[2]+rect[0])/2) and (self.e5[1]<=(rect[3]+rect[1])/2 and self.e5[1] + self.e5[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e5")
			return True;
		else :
			return False;
	def crossS5(self,rect):
		if((self.s5[0]<=(rect[2]+rect[0])/2 and self.s5[0] + self.s5[2] >= (rect[2]+rect[0])/2) and (self.s5[1]<=(rect[3]+rect[1])/2 and self.s5[1] + self.s5[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s5")
			return True;
		else :
			return False;


	# Draw the predicted bounding box
	def drawPred(self, classId, conf, left, top, right, bottom):
		# Draw a bounding box.
		left = int(left)
		top = int(top)
		right =int(right)
		bottom = int(bottom)
		cv2.rectangle(self.frame, (int(left), int(top)), (int(right), int(bottom)), (255, 178, 50), 3) 
		# Get the label for the class name and its confidence
		rect = (left, top, right, bottom)
		label = classId
		print(label)
		if (self.crossE1(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0],rect[1] ,(rect[2]-rect[0]),(rect[3]-rect[1])))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e1")
			
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1
		elif (self.crossE2(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e2")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1
		elif (self.crossE3(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e3")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1

		elif (self.crossE4(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e4")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1

		elif (self.crossE5(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e5")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1

			

# Remove the bounding boxes with low confidence using non-maxima suppression
	def postprocess(self, results):
		frameHeight = self.frame.shape[0]
		frameWidth = self.frame.shape[1]

		for cat, score, bounds in results:
			center_x, center_y, width, height = bounds
			left = int(center_x - width / 2)
			top = int(center_y - height / 2)
			self.drawPred(cat, score, left, top, left + width, top + height)
        		#cv.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
        		#cv.putText(frame,str(cat.decode("utf-8")),(int(x),int(y)),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0))

