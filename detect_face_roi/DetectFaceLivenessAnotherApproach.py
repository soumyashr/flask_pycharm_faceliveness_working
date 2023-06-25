# -*- coding: utf-8 -*-
"""
@Ref : dlib library
"""
import math

# import the necessary packages
from scipy.spatial import distance as dist
# from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
# import numpy as np
# import argparse
import imutils
import time
import dlib
import cv2

from detect_face_roi.Constants import Constants


class DetectFaceLivenessAnotherApproach:
	const = Constants()
	# initialize the frame counters and the total number of blinks
	COUNTER = 0
	TOTAL = 0
	# Eye blink threshold to pass the test, ie how many times eyes must be blinking in given DETECTION_WINDOW
	BLINK_COUNT_THRESH = math.trunc(self.const.DETECTION_WINDOW / 4)

	def __init__(self, model_name):
		print('DetectFaceLivenessAnotherApproach being invoked')
		self.model_name = model_name

	def eye_aspect_ratio(self, eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])
		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])
		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)
		# return the eye aspect ratio
		return ear

	args = {
		"shape_predictor": "./model/shape_predictor_68_face_landmarks.dat",
		"image": "images/example_Som 11-01-23.png"
	}

	def initialise(self):

		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		print("[INFO] loading facial landmark predictor...")
		# print(os.getcwd())
		detector = dlib.get_frontal_face_detector()
		# predictor = dlib.shape_predictor(args["shape_predictor"])
		predictor = dlib.shape_predictor(self.model_name)

		return detector, predictor

	def capture_video_stream(self):
		# start the video stream thread
		print("[INFO] starting video stream thread...")

		# Below is the case when a file stream is being read
		# fileStream = True
		# vs = FileVideoStream(args["video"]).start()

		# Below is the case when a live webcam is being used
		fileStream = False
		vs = VideoStream(src=0).start() # src=0 for Windows
		# vs = VideoStream(src=-1).start() # -1 for linux

		# vs = VideoStream(usePiCamera=True).start()

		time.sleep(1.0)
		return vs, fileStream

	def count_eye_blinks(self):

		detector, predictor = self.initialise()

		vs, fileStream = self.capture_video_stream()
		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# loop over frames from the video stream

		t_end = time.time() + self.const.DETECTION_WINDOW
		start_time = time.time()

		while time.time() < t_end:

		#while True:
			# if this is a file video stream, then we need to check if
			# there any more frames left in the buffer to process
			if fileStream and not vs.more():
				break
			# grab the frame from the threaded video file stream, resize
			# it, and convert it to grayscale
			# channels)
			frame = vs.read()
			frame = imutils.resize(frame, width=450)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# detect faces in the grayscale frame
			detected = detector(gray, 0)

			# loop over the face detections
			for rect in detected:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = self.eye_aspect_ratio(leftEye)
				rightEAR = self.eye_aspect_ratio(rightEye)
				# average the eye aspect ratio together for both eyes
				ear = (leftEAR + rightEAR) / 2.0

				print('At fraction of time:', time.time() -  start_time , '  - ear:', ear)

				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				# if calculated eye aspect ratio is lesser than the threshold, then record an eye-blink event. increment the blink frame counter
				# if ear < self.const.EYE_AR_THRESH:
				if ear < self.const.EYE_AR_THRESH:
					self.COUNTER += 1
				# otherwise, the eye aspect ratio is not below the blink threshold, so eyes did not blink
				else:
					# if the eyes were closed for a sufficient number of times then increment the total number of blinks
					# ie if sufficient no of consecutive frames contained an eye blink ratio below pre-defined threshold
					if self.COUNTER >= self.const.EYE_AR_CONSEC_FRAMES:
						self.TOTAL += 1
					# reset the eye frame counter
					self.COUNTER = 0

				# draw the total number of blinks on the frame along with
				# the computed eye aspect ratio for the frame
				cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# show the frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()

		# If TOTAL value is not more than our threshold ie eyes did not blink sufficient number of times, raise alarm
		print('Count of eye-blinks : ', self.TOTAL)
		print('BLINK_COUNT_THRESH : ', self.BLINK_COUNT_THRESH)
		if self.TOTAL < self.BLINK_COUNT_THRESH:
			print('Raise alarm')
			Liveness_Detected = False
		else :
			Liveness_Detected = True
		time_taken = time.time() - start_time
		exec_time = '%.2f'%time_taken
		print('[INFO] execution  time {} seconds'.format(exec_time))

		return Liveness_Detected


