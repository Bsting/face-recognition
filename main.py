from mtcnn import MTCNN
from faceBoxes import Faceboxes
from faceEngine import FaceEngine
from photoBoothApp import PhotoBoothApp
from imutils.video import VideoStream
import threading
import numpy as np

if __name__ == '__main__':
    try:
        print("Starting...")
        nameIdentity = False
        mtcnn = MTCNN()
        faceboxes = Faceboxes()
        faceEngine = FaceEngine()
        faceEngine.load_state('pretrained_model/model.pth')
        
        print('Face engine loaded.')

        print("Loading facebank...")
        targets, names = faceEngine.load_facebank()
        target_count = len(targets)
        print('Facebank loaded - records: {0}'.format(target_count))
        print("Starting camera...")
        vs = VideoStream(0).start()
        
        pba = PhotoBoothApp(vs, 'data', mtcnn, faceboxes, faceEngine, targets, names, target_count, nameIdentity)

        thread = threading.Thread(target=pba.faceCaptureTask, args=())
        thread.daemon = True
        thread.start()    

        thread2 = threading.Thread(target=pba.faceIdentifyTask, args=())
        thread2.daemon = True
        thread2.start()

        pba.root.mainloop()
    except RuntimeError as e:
        print("RuntimeError: {0}".format(e))
        exit(-1)