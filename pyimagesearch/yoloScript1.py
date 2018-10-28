# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from PIL import Image
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

class PhotoBoothApp:
	def __init__(self,trackerType):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.videoPath=None
		self.isPlay=False
		self.isPause=False
		self.isRestart=False

		
		# Initialize YOLO Detection parameters
		self.confThreshold = 0.4  #Confidence threshold
		self.nmsThreshold = 0.4   #Non-maximum suppression threshold
		self.inpWidth = 416       #Width of network's input image
		self.inpHeight = 416      #Height of network's input image
		
		self.sec = 0
		self.frameRate = 0.1
		# Load names of classes
		self.classesFile = "coco.names";
		self.classes = None
		with open(self.classesFile, 'rt') as f:
		    self.classes = f.read().rstrip('\n').split('\n')

		# Give the configuration and weight files for the model and load the network using them.
		self.modelConfiguration = "yolov3.cfg";
		self.modelWeights = "yolov3.weights";

		self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
		
		

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
		self.e1 = (0, 0, 0, 0)
		self.e2 = (0, 0, 0, 0)
		self.e3 = (0, 0, 0, 0)
		self.s1 = (0, 0, 0, 0)
		self.s2 = (0, 0, 0, 0)
		self.s3 = (0, 0, 0, 0)
		self.frame = None
		self.crop_frame = None
		self.thread = None
		self.stopEvent = None

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None
		
		# initialize OpenCV's special multi-object tracker
		self.trackers = cv2.MultiTracker_create()         
		self.trackerType = trackerType 

		btnDestroy = tki.Button(self.root, text="Cancel",command=self.root.destroy)
		btnDestroy.pack()
		btnDestroy.place(height=50, width=150, x=10, y=100)		
	
		btnRestart = tki.Button(self.root, text="Restart",command=self.restart)
		btnRestart.pack()
		btnRestart.place(height=50, width=150, x=170, y=100)

		btnVideo = tki.Button(self.root, text="Uploade your Video",command=self.selectFile)
		btnVideo.pack()
		btnVideo.place(height=50, width=310, x=10, y=200)
		

		btnPlay = tki.Button(self.root, text="Play",command=self.play)
		btnPlay.pack()
		btnPlay.place(height=50, width=150, x=10, y=300)

		btnPause = tki.Button(self.root, text="Pause",command=self.pause)
		btnPause.pack()
		btnPause.place(height=50, width=150, x=170, y=300)
		
		
		# create a button, that when pressed, will take the current
		# frame and save it to file
		btnE1 = tki.Button(self.root, text="Draw E1",command=self.drawE1)
		btnE1.pack()
		btnE1.place(height=50, width=150, x=10, y=400)
		btnS1 = tki.Button(self.root, text="Draw S1",command=self.drawS1)
		btnS1.pack()
		btnS1.place(height=50, width=150, x=170, y=400)
		
		btnE2 = tki.Button(self.root, text="Draw E2",command=self.drawE2)
		btnE2.pack()
		btnE2.place(height=50, width=150, x=10, y=500)
		btnS2 = tki.Button(self.root, text="Draw S2",command=self.drawS2)
		btnS2.pack()
		btnS2.place(height=50, width=150, x=170, y=500)

		btnE3 = tki.Button(self.root, text="Draw E3",command=self.drawE3)
		btnE3.pack()
		btnE3.place(height=50, width=150, x=10, y=600)
		btnS3 = tki.Button(self.root, text="Draw S3",command=self.drawS3)
		btnS3.pack()
		btnS3.place(height=50, width=150, x=170, y=600)
		

		# start a thread that constantly pools the video sensor for
		# the most recently read frame	
				
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("Vehicles Tracking")
		self.root.geometry("1200x800")
		#self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

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
				if self.isPlay:
					# grab the frame from the video stream and resize it to
					# have a maximum width of 800 pixels
					self.sec=self.sec+self.frameRate
					self.vs.set(cv2.CAP_PROP_POS_MSEC,self.sec*1000)
					self.frame = self.vs.read()
					self.frame = self.frame[1]
					self.frame = imutils.resize(self.frame, width=800)
					self.frameRate = 0.1
					(success, boxes) = self.trackers.update(self.frame)
					cv2.rectangle(self.frame, (self.e1[0], self.e1[1]), (self.e1[0] + self.e1[2], self.e1[1] + self.e1[3]), (255, 0, 0), 2)
					cv2.rectangle(self.frame, (self.s1[0], self.s1[1]), (self.s1[0] + self.s1[2], self.s1[1] + self.s1[3]), (0, 255, 0), 2)
					cv2.rectangle(self.frame, (self.e2[0], self.e2[1]), (self.e2[0] + self.e3[2], self.e2[1] + self.e2[3]), (255, 0, 0), 2)
					cv2.rectangle(self.frame, (self.s2[0], self.s2[1]), (self.s2[0] + self.s2[2], self.s2[1] + self.s2[3]), (0, 255, 0), 2)
					cv2.rectangle(self.frame, (self.e3[0], self.e3[1]), (self.e3[0] + self.e3[2], self.e3[1] + self.e3[3]), (255, 0, 0), 2)
					cv2.rectangle(self.frame, (self.s3[0], self.s3[1]), (self.s3[0] + self.s3[2], self.s3[1] + self.s3[3]), (0, 255, 0), 2)
					for box in boxes:
						(x, y, w, h) = [int(v) for v in box]
						cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
					
					if self.e1[3] == 0 :
						crop_frame = self.frame[self.e1[1]:self.e1[1] + self.e1[3], self.e1[0]:self.e1[0] + self.e1[2]]
						#crop_frame = self.frame						
						# Create a 4D blob from a frame.
						blob = cv2.dnn.blobFromImage(crop_frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)

						# Sets the input to the network
						self.net.setInput(blob)
						# Runs the forward pass to get output of the output layers
						outs = self.net.forward(self.getOutputsNames())
						# Remove the bounding boxes with low confidence
						self.postprocess(crop_frame, outs)
						#Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of 						the layers(in layersTimes)
						#t, _ = self.net.getPerfProfile()
						#label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
						#cv2.putText(crop_frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
						
					

						
					# OpenCV represents images in BGR order; however PIL
					# represents images in RGB order, so we need to swap
					# the channels, then convert to PIL and ImageTk format
					image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
					image = Image.fromarray(image)
					image = ImageTk.PhotoImage(image)
								# if the panel is not None, we need to initialize it
					if self.panel is None:
						self.panel = tki.Label(image=image)
						self.panel.image = image
						self.panel.pack()
						self.panel.place(x=350, y=200)
								# otherwise, simply update the panel
					else:
						self.panel.configure(image=image)
						self.panel.image = image
		except RuntimeError as e:
			print("[INFO] caught a RuntimeError")

	def selectFile(self):
		self.videoPath =  tki.filedialog.askopenfilename(initialdir = "",title = "Select file",filetypes = (("All files","*.*"),("mp4 files","*.mp4")))
		self.vs = cv2.VideoCapture(self.videoPath)
		#self.vs.set(cv2.CAP_PROP_POS_MSEC,self.sec*1000)
		self.frame = self.vs.read()
		self.frame = self.frame[1]
		self.frame = imutils.resize(self.frame, width=800)
		# OpenCV represents images in BGR order; however PIL
		# represents images in RGB order, so we need to swap
		# the channels, then convert to PIL and ImageTk format
		image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
		# if the panel is not None, we need to initialize it
		if self.panel is None:
			self.panel = tki.Label(image=image)
			self.panel.image = image
			self.panel.pack()
			self.panel.place(x=350, y=200)
		# otherwise, simply update the panel
		else:
			self.panel.configure(image=image)
			self.panel.image = image		
	def restart(self):
		self.isRestart=True
		self.isPlay = False
		self.isPause = False
		self.e1 = (0, 0, 0, 0)
		self.e2 = (0, 0, 0, 0)
		self.e3 = (0, 0, 0, 0)
		self.s1 = (0, 0, 0, 0)
		self.s2 = (0, 0, 0, 0)
		self.s3 = (0, 0, 0, 0)

		
		# initialize OpenCV's special multi-object tracker
		self.trackers = cv2.MultiTracker_create() 
        
	def play(self):
		self.isRestart=False
		self.isPlay = True
		self.isPause = False
	def pause(self):
		self.isPause = True
		self.isPlay = False		
	def drawE1(self):
		self.e1 = cv2.selectROI("Select E1",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
	def drawS1(self):
		self.s1 = cv2.selectROI("Select S1",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
	def drawE2(self):
		self.e2 = cv2.selectROI("Select E2",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
	def drawS2(self):
		self.s2 = cv2.selectROI("Select S2",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
	def drawE3(self):
		self.e3 = cv2.selectROI("Select E3",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
	def drawS3(self):
		self.s3 = cv2.selectROI("Select S3",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)

	# Get the names of the output layers
	def getOutputsNames(self):
	    # Get the names of all the layers in the network
	    layersNames = self.net.getLayerNames()
	    # Get the names of the output layers, i.e. the layers with unconnected outputs
	    return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	# Draw the predicted bounding box
	def drawPred(self, crop_frame, classId, conf, left, top, right, bottom):
		# Draw a bounding box.
		cv2.rectangle(crop_frame, (left, top), (right, bottom), (255, 178, 50), 3)  
		label = '%.2f' % conf
		
		# Get the label for the class name and its confidence
		if self.classes:
			assert(classId < len(self.classes))
			label = '%s:%s' % (self.classes[classId], label)

		#Display the label at the top of the bounding box
		labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		top = max(top, labelSize[1])
		tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
		cv2.rectangle(crop_frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
		self.trackers.add(tracker, self.frame, (self.e1[0]+left, self.e1[1]+top - round(1.5*labelSize[1]),labelSize[0], labelSize[0]))
		cv2.putText(crop_frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
		#self.frameRate = 0.6

# Remove the bounding boxes with low confidence using non-maxima suppression
	def postprocess(self,crop_frame, outs):
		frameHeight = crop_frame.shape[0]
		frameWidth = crop_frame.shape[1]

		classIds = []
		confidences = []
		boxes = []
		# Scan through all the bounding boxes output from the network and keep only the
		# ones with high confidence scores. Assign the box's class label as the class with the highest score.
		classIds = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				classId = np.argmax(scores)
				confidence = scores[classId]
				if confidence > self.confThreshold and classId == 2:
					center_x = int(detection[0] * frameWidth)
					center_y = int(detection[1] * frameHeight)
					width = int(detection[2] * frameWidth)
					height = int(detection[3] * frameHeight)
					left = int(center_x - width / 2)
					top = int(center_y - height / 2)
					classIds.append(classId)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])

	    # Perform non maximum suppression to eliminate redundant overlapping boxes with
	    # lower confidences.
		indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
		for i in indices:
			i = i[0]
			box = boxes[i]
			left = box[0]
			top = box[1]
			width = box[2]
			height = box[3]
			print(classIds[i])
			self.drawPred(crop_frame, classIds[i], confidences[i], left, top, left + width, top + height)


	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		#self.stopEvent.set()
		#self.vs.stop()
		self.root.quit()
