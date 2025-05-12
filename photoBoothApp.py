# import the necessary packages
from __future__ import print_function
from PIL import Image, ImageFont, ImageDraw
from PIL import ImageTk
from faceEngine import FaceEngine
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import math
import os
import time
import torch
import numpy as np
import io

class PhotoBoothApp:
	def __init__(self, videoStream, dataPath, mtcnn, faceboxes, faceEngine, targets, names, targetCount, nameIdentity):
		self.videoStream = videoStream
		self.dataPath = dataPath
		self.blurScore = 30
		self.frame = None
		self.thread = None
		self.stopEvent = None
		self.mtcnn = mtcnn
		self.faceboxes = faceboxes
		self.faceEngine = faceEngine
		self.targets = targets
		self.names = names
		self.targetCount = targetCount
		self.nameIdentity = nameIdentity
		self.imageUnknown = Image.open('images/unknown.jpg')
		
		self.root = tki.Tk()
		self.root.configure(background='black')
		self.windowHeight = 650
		self.windowWidth = 1380
		# Positions the window 50 pixels from the left and 50 pixels from the top of the screen.
		self.root.geometry(f"{self.windowWidth}x{self.windowHeight}+50+50") 

		marginX = 10
		marginY = 55
		marginGap = 5
		registerIdentity = tki.Label(self.root, bg="black", fg="white", font=('times', 24), anchor='w', text=f'Registered Identity: {self.targetCount}')
		registerIdentity.place(x=marginX, y=10)

		self.cameraViewWidth = 850
		self.cameraViewHeight = 590
		self.cameraView = tki.Label(self.root)
		self.cameraView.place(x=marginX, y=marginY, width=self.cameraViewWidth, height=self.cameraViewHeight)

		imageWidth = 112
		resultHeight = 112
		textWidth = 120
		result_col_1_x = self.cameraViewWidth + 20
		result_col_2_x = result_col_1_x + imageWidth + marginX
		result_col_3_x = result_col_2_x + imageWidth + marginX
		result_col_4_x = result_col_3_x + textWidth + marginX

		col_1_title = tki.Label(self.root, bg="black", fg="white", font=('times', 22), anchor='w', text='Capture')
		col_1_title.place(x=result_col_1_x, y=10)
		col_2_title = tki.Label(self.root, bg="black", fg="white", font=('times', 22), anchor='w', text='Detect')
		col_2_title.place(x=result_col_2_x, y=10)
		col_3_title = tki.Label(self.root, bg="black", fg="white", font=('times', 22), anchor='w', text='Time')
		col_3_title.place(x=result_col_3_x, y=10)
		col_4_title = tki.Label(self.root, bg="black", fg="white", font=('times', 22), anchor='w', text='Score')
		col_4_title.place(x=result_col_4_x, y=10)

		self.target_1 = tki.Label(self.root, bg="blue")
		self.target_1.place(x=result_col_1_x, y=marginY, width=imageWidth, height=resultHeight)
		self.subject_1 = tki.Label(self.root, bg="blue", fg="white",  font=('times', 20), wraplength=imageWidth, anchor='w')
		self.subject_1.place(x=result_col_2_x, y=marginY, width=imageWidth, height=resultHeight)
		self.time_1 = tki.Label(self.root, bg="blue", fg="white", font=('times', 18), wraplength=imageWidth, anchor='nw')
		self.time_1.place(x=result_col_3_x, y=marginY, width=textWidth, height=resultHeight)
		self.score_1 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='nw')
		self.score_1.place(x=result_col_4_x, y=marginY, width=textWidth, height=resultHeight)

		target_2_y = marginY + resultHeight + marginGap
		self.target_2 = tki.Label(self.root, bg="blue")
		self.target_2.place(x=result_col_1_x, y=target_2_y, width=imageWidth, height=resultHeight)
		self.subject_2 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='w')
		self.subject_2.place(x=result_col_2_x, y=target_2_y, width=imageWidth, height=resultHeight)
		self.time_2 = tki.Label(self.root, bg="blue", fg="white", font=('times', 18), wraplength=imageWidth, anchor='nw')
		self.time_2.place(x=result_col_3_x, y=target_2_y, width=textWidth, height=resultHeight)
		self.score_2 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='nw')
		self.score_2.place(x=result_col_4_x, y=target_2_y, width=textWidth, height=resultHeight)

		target_3_y = target_2_y + resultHeight + marginGap
		self.target_3 = tki.Label(self.root, bg="blue")
		self.target_3.place(x=result_col_1_x, y=target_3_y, width=imageWidth, height=resultHeight)
		self.subject_3 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='w')
		self.subject_3.place(x=result_col_2_x, y=target_3_y, width=imageWidth, height=resultHeight)
		self.time_3 = tki.Label(self.root, bg="blue", fg="white", font=('times', 18), wraplength=imageWidth, anchor='nw')
		self.time_3.place(x=result_col_3_x, y=target_3_y, width=textWidth, height=resultHeight)
		self.score_3 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='nw')
		self.score_3.place(x=result_col_4_x, y=target_3_y, width=textWidth, height=resultHeight)

		target_4_y = target_3_y + resultHeight + marginGap
		self.target_4 = tki.Label(self.root, bg="blue")
		self.target_4.place(x=result_col_1_x, y=target_4_y, width=imageWidth, height=resultHeight)
		self.subject_4 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='w')
		self.subject_4.place(x=result_col_2_x, y=target_4_y, width=imageWidth, height=resultHeight)
		self.time_4 = tki.Label(self.root, bg="blue", fg="white", font=('times', 18), wraplength=imageWidth, anchor='nw')
		self.time_4.place(x=result_col_3_x, y=target_4_y, width=textWidth, height=resultHeight)
		self.score_4 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='nw')
		self.score_4.place(x=result_col_4_x, y=target_4_y, width=textWidth, height=resultHeight)

		target_5_y = target_4_y + resultHeight + marginGap
		self.target_5 = tki.Label(self.root, bg="blue")
		self.target_5.place(x=result_col_1_x, y=target_5_y, width=imageWidth, height=resultHeight)
		self.subject_5 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='w')
		self.subject_5.place(x=result_col_2_x, y=target_5_y, width=imageWidth, height=resultHeight)
		self.time_5 = tki.Label(self.root, bg="blue", fg="white", font=('times', 18), wraplength=imageWidth, anchor='nw')
		self.time_5.place(x=result_col_3_x, y=target_5_y, width=textWidth, height=resultHeight)
		self.score_5 = tki.Label(self.root, bg="blue", fg="white", font=('times', 20), wraplength=imageWidth, anchor='nw')
		self.score_5.place(x=result_col_4_x, y=target_5_y, width=textWidth, height=resultHeight)

		self.lock = threading.Lock()
		self.stopEvent = threading.Event()
		self.root.wm_title("Applied Deep Learning - Capstone Project: Attendance System with Face Recognition")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

	def faceCaptureTask(self):
		try:
			torch.set_num_threads(1)
			start_time = time.time()
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				try:
					self.frame = self.videoStream.read()
					if self.frame is None:
						continue
					# grab the frame from the video stream and resize it
					self.frame = cv2.resize(self.frame, (self.cameraViewWidth, self.cameraViewHeight))
					# OpenCV represents images in BGR order; however PIL
					# represents images in RGB order, so we need to swap
					# the channels, then convert to PIL and ImageTk format
					image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
					image = Image.fromarray(image)
						
					delta_time = int((time.time() - start_time) * 1000)
					if delta_time > 5:
						image = self.detectFace(image, True)
						start_time = time.time()
						
					image = ImageTk.PhotoImage(image)
					self.cameraView.configure(image=image)
					self.cameraView.image = image
				except TypeError as e:
					print("faceCaptureTask TypeError: {0}".format(e))
		except RuntimeError as e:
			print("faceCaptureTask RuntimeError: {0}".format(e))
			self.stopEvent.set()
		self.root.destroy()

	def faceIdentifyTask(self):
		try:
			torch.set_num_threads(1)
			start_time = time.time()
			result_idx = 0
			while not self.stopEvent.is_set():
				try:
					frame = self.videoStream.read()
					if frame is None:
						continue
										
					delta_time = int((time.time() - start_time) * 1000)
					if delta_time < 500:
						continue
					
					min_dimension_size = 590
					frame_height, frame_width = frame.shape[:2]			
					if frame_width > min_dimension_size and frame_height > min_dimension_size:
						if frame_width > min_dimension_size:
							frame = imutils.resize(frame, width=min_dimension_size)
						else:
							frame = imutils.resize(frame, height=min_dimension_size)

					min_length = min(frame.shape[:2])
					min_face_size = (int)(min_length / 10)
					
					image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					image = Image.fromarray(image)
					faces, bboxes, results, scores, blurScores = self.identifyFace(image, min_face_size=min_face_size)
					if results is None or len(results) < 1:
						continue
				
					for i, result in enumerate(results):
						if result == (-1):
							writeLog(f"Name: UNKNOWN, Blur: {blurScores[i]}, Score: {scores[i]}")
							if self.nameIdentity == True:								
								self.setResultWithName(result_idx, faces[i], 'UNKNOWN', '-', 'red')
							else:
								self.setResultWithPhoto(result_idx, faces[i], self.imageUnknown, '-', 'red')
						else:
							writeLog(f"Name: {self.names[result]}, , Blur: {blurScores[i]}, Score: {scores[i]}")
							if self.nameIdentity == True:								
								self.setResultWithName(result_idx, faces[i], self.names[result], '{:.2f}'.format(scores[i]), 'white')
							else:
								detectedFace = self.getFace(self.names[result])
								if detectedFace is not None:
									detectedFace = detectedFace.resize((112, 112), resample=Image.ANTIALIAS)
									self.setResultWithPhoto(result_idx, faces[i], detectedFace, '{:.2f}'.format(scores[i]), 'white')
					
						result_idx = result_idx + 1
						if result_idx > 3:
							result_idx = 0			
				except TypeError as e:
					writeLog(f"faceIdentifyTask TypeError: {e}")
					start_time = time.time()
		except RuntimeError as e:
			writeLog(f"faceIdentifyTask RuntimeError: {e}")
			self.stopEvent.set()
			
		self.root.destroy()

	def setResultWithName(self, index, face, name, score, textColor):
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

		if hasattr(self.target_4, 'image'):
			self.target_5.configure(bg='black', image=self.target_4.image)
			self.target_5.image = self.target_4.image
			self.subject_5.configure(bg='black', fg=self.subject_4.cget("foreground"), text=self.subject_4.cget("text"), font=self.subject_4.cget("font"))
			self.time_5.configure(bg='black', fg=self.time_4.cget("foreground"), text=self.time_4.cget("text"))
			self.score_5.configure(bg='black', fg=self.score_4.cget("foreground"), text=self.score_4.cget("text"))

		if hasattr(self.target_3, 'image'):
			self.target_4.configure(bg='black', image=self.target_3.image)
			self.target_4.image = self.target_3.image
			self.subject_4.configure(bg='black', fg=self.subject_3.cget("foreground"), text=self.subject_3.cget("text"), font=self.subject_3.cget("font"))
			self.time_4.configure(bg='black', fg=self.time_3.cget("foreground"), text=self.time_3.cget("text"))
			self.score_4.configure(bg='black', fg=self.score_3.cget("foreground"), text=self.score_3.cget("text"))

		if hasattr(self.target_2, 'image'):
			self.target_3.configure(bg='black', image=self.target_2.image)
			self.target_3.image = self.target_2.image
			self.subject_3.configure(bg='black', fg=self.subject_2.cget("foreground"), text=self.subject_2.cget("text"), font=self.subject_2.cget("font"))
			self.time_3.configure(bg='black', fg=self.time_2.cget("foreground"), text=self.time_2.cget("text"))
			self.score_3.configure(bg='black', fg=self.score_2.cget("foreground"), text=self.score_2.cget("text"))

		if hasattr(self.target_1, 'image'):
			self.target_2.configure(bg='black', image=self.target_1.image)
			self.target_2.image = self.target_1.image
			self.subject_2.configure(bg='black', fg=self.subject_1.cget("foreground"), text=self.subject_1.cget("text"), font=self.subject_1.cget("font"))
			self.time_2.configure(bg='black', fg=self.time_1.cget("foreground"), text=self.time_1.cget("text"))
			self.score_2.configure(bg='black', fg=self.score_1.cget("foreground"), text=self.score_1.cget("text"))

		image = ImageTk.PhotoImage(face)
		self.target_1.configure(image=image)
		self.target_1.image = image
		subjectFontSize = getFontSize(len(name))
		self.subject_1.configure(bg='black', fg=textColor, text=name, font=('times', subjectFontSize))
		self.time_1.configure(bg='black', fg=textColor, text=timestamp)
		self.score_1.configure(bg='black', fg=textColor, text=score)			

	def setResultWithPhoto(self, index, targetFace, subjectFace, score, textColor):
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
		
		if hasattr(self.target_4, 'image'):
			self.target_5.configure(bg='black', image=self.target_4.image)
			self.target_5.image = self.target_4.image
			self.subject_5.configure(bg='black', image=self.subject_4.image)
			self.subject_5.image = self.subject_4.image
			self.time_5.configure(bg='black', fg=self.time_4.cget("foreground"), text=self.time_4.cget("text"))
			self.score_5.configure(bg='black', fg=self.score_4.cget("foreground"), text=self.score_4.cget("text"))

		if hasattr(self.target_3, 'image'):
			self.target_4.configure(bg='black', image=self.target_3.image)
			self.target_4.image = self.target_3.image
			self.subject_4.configure(bg='black', image=self.subject_3.image)
			self.subject_4.image = self.subject_3.image
			self.time_4.configure(bg='black', fg=self.time_3.cget("foreground"), text=self.time_3.cget("text"))
			self.score_4.configure(bg='black', fg=self.score_3.cget("foreground"), text=self.score_3.cget("text"))

		if hasattr(self.target_2, 'image'):
			self.target_3.configure(bg='black', image=self.target_2.image)
			self.target_3.image = self.target_2.image
			self.subject_3.configure(bg='black', image=self.subject_2.image)
			self.subject_3.image = self.subject_2.image
			self.time_3.configure(bg='black', fg=self.time_2.cget("foreground"), text=self.time_2.cget("text"))
			self.score_3.configure(bg='black', fg=self.score_2.cget("foreground"), text=self.score_2.cget("text"))

		if hasattr(self.target_1, 'image'):
			self.target_2.configure(bg='black', image=self.target_1.image)
			self.target_2.image = self.target_1.image
			self.subject_2.configure(bg='black', image=self.subject_1.image)
			self.subject_2.image = self.subject_1.image
			self.time_2.configure(bg='black', fg=self.time_1.cget("foreground"), text=self.time_1.cget("text"))
			self.score_2.configure(bg='black', fg=self.score_1.cget("foreground"), text=self.score_1.cget("text"))

		targetImage = ImageTk.PhotoImage(targetFace)
		self.target_1.configure(bg='black', image=targetImage)
		self.target_1.image = targetImage
		
		subjectImage = ImageTk.PhotoImage(subjectFace)
		self.subject_1.configure(bg='black', image=subjectImage)
		self.subject_1.image = subjectImage

		self.time_1.configure(bg='black', fg=textColor, text=timestamp)
		self.score_1.configure(bg='black', fg=textColor, text=score)

	def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.dataPath, filename))

		# save the file
		cv2.imwrite(p, self.frame.copy())
		writeLog("[INFO] saved {}".format(filename))

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		writeLog("[INFO] closing...")
		self.stopEvent.set()
		self.videoStream.stop()
		self.root.quit()
		#self.root.destroy()

	def drawBoxName(self, bbox, name, image, color=(0, 255, 0)):
		draw = ImageDraw.Draw(image)
		draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline=color, width=5)
		if len(name) > 0:
			font = ImageFont.truetype("arial.ttf", 20)		
			draw.text((bbox[0] + 5, bbox[1] - 25), name, fill=color, width=3, font=font)
		return image
	
	def draw_box_name2(self, bbox, landmark, text1, text2, text3, image, color=(0, 255, 0)):
		draw = ImageDraw.Draw(image)
		draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline=color, width=5)
		font = ImageFont.truetype("arial.ttf", 20)	
		if len(text1) > 0:				
			draw.text((bbox[0] + 5, bbox[1] - 20), text1, fill=(0, 255, 0), width=3, font=font)
		if len(text2) > 0:
			draw.text((bbox[0] + 5, bbox[1] - 40), text2, fill=(0, 255, 0), width=3, font=font)
		if len(text3) > 0:
			draw.text((bbox[0] + 5, bbox[1] - 60), text3, fill=(0, 255, 0), width=3, font=font)
		
		for i in range(5):
			draw.ellipse([(landmark[i] - 1.0, landmark[i + 5] - 1.0),
					(landmark[i] + 1.0, landmark[i + 5] + 1.0)], outline='red', width=3)                
		return image

	def drawLandmark(self, image, landmark, color=(255, 0, 0)):
		draw = ImageDraw.Draw(image)
		for i in range(5):
			draw.ellipse([(landmark[i] - 1.0, landmark[i + 5] - 1.0),
					(landmark[i] + 1.0, landmark[i + 5] + 1.0)], outline=color, width=3)
		return image

	def identifyFace(self, image, min_face_size=50):
		try:
			largest_face = False
			if largest_face:
				bbox, face, _ = self.mtcnn.align_largest_bbox(
					image, 
					min_face_size, 
					thresholds=[0.80, 0.90, 0.98])
				if face is None:
					return None, None, None, None, None
				bbox = bbox.astype(int)
				valid_bboxes = []
				valid_faces = []
				blur_scores = []
				start_time = time.time()
				cv_face = np.array(face)
				blur_score = getBlurScore(cv_face[:, :, ::-1].copy())
				if blur_score > 45.0:
					valid_faces.append(face)
					valid_bboxes.append(bbox)
					blur_scores.append(blur_score)
				else:
					writeLog(f"Rejected image, blur: {blur_score}")
					return None, None, None, None, None		
				if len(valid_faces) > 0:
					results, scores = self.faceEngine.infer(valid_faces, self.targets[:self.targetCount,])
					writeLog(f'Identification time: {int((time.time() - start_time) * 1000)} ms')
					return valid_faces, valid_bboxes, results, scores, blur_scores
			else:
				bboxes, faces, _ = self.mtcnn.align_multi(
					image, 
					10, 
					min_face_size, 
					thresholds=[0.80, 0.90, 0.98])
				if bboxes is None or len(bboxes) == 0:
					return None, None, None, None, None
				bboxes = bboxes.astype(int)
				valid_bboxes = []
				valid_faces = []
				blur_scores = []
				start_time = time.time()
				for i, face in enumerate(faces):
					cv_face = np.array(face)
					blur_score = getBlurScore(cv_face[:, :, ::-1].copy())
					if blur_score > self.blurScore:
						valid_faces.append(face)
						valid_bboxes.append(bboxes[i])
						blur_scores.append(blur_score)
					else:
						writeLog(f"Rejected image, blur: {blur_score}")
				if len(valid_faces) > 0:
					results, scores = self.faceEngine.infer(valid_faces, self.targets[:self.targetCount,])
					writeLog(f'Identification time: {int((time.time() - start_time) * 1000)} ms')
					return valid_faces, valid_bboxes, results, scores, blur_scores
		except Exception as e:
			writeLog('Error detected:',e)
		return None, None, None, None, None
	
	def detectFace(self, image, useMtcnn=True):
		orig_image = image.copy()
		try:
			if useMtcnn:
				new_width = 420
				orig_width, orig_height = image.size
				scale = new_width / orig_width
				new_height = orig_height * scale
				new_size = int(math.floor(new_width)), int(math.floor(new_height))
				image = image.resize(new_size, resample=Image.ANTIALIAS)
				#image.thumbnail(new_size, Image.ANTIALIAS)
			else:
				scale = 1
			#detect_start = time.time()
			if useMtcnn:
				bboxes, faces, landmarks = self.mtcnn.align_multi(
					image, 
					5, 
					min_face_size=35.0, 
					thresholds=[0.80, 0.90, 0.95])
			else:
				bboxes = self.faceboxes.detect_faces(image, resize=1, confidence_threshold=0.8)				
			#writeLog('detect face time: {0} ms'.format(int((time.time() - detect_start) * 1000)))
			if bboxes is None or len(bboxes) == 0:
				return orig_image
			bboxes = bboxes / scale
			bboxes = bboxes.astype(int)
			landmarks = landmarks / scale
			for bbox in bboxes:
				orig_image = self.drawBoxName(bbox, "", orig_image, (0, 0, 255))					
		except Exception as e:
			writeLog('Error detected:',e)		
		return orig_image

	def getFace(self, name):
		try:
			file = os.path.join(self.faceEngine.conf.facebank_path, 'dataset', f'{name}.jpg')
			image = Image.open(file)
			_, face, landmark = self.mtcnn.align_largest_bbox(
				image,
				65, 
				thresholds=[0.80, 0.95, 0.99])
		except Exception as e:
			writeLog('Get face error:',e)
			return None
		return face
	
def getFontSize(textLength):
	fontSize = 25
	adjust = int(textLength / 3)
	fontSize = fontSize - adjust
	if fontSize > 16:
		fontSize = 16
	elif fontSize < 9:
		fontSize = 9
	return fontSize

def getBlurScore(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian	
	# remove noise by blurring with a Gaussian filter
	imageBlur = cv2.GaussianBlur(image, (3,3), 0)
	imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
	return cv2.Laplacian(imageGray, cv2.CV_64F).var()

def writeLog(message, log_file='app.log'):
    print(message)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
    with open(log_file, 'a') as file:  # Open file in append mode
        file.write(f'{timestamp} - {message}\n')