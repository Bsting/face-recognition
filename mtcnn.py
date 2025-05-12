import numpy as np
import torch
from PIL import Image
from face_detector.mtcnn.get_nets import PNet, RNet, ONet
from face_detector.mtcnn.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, correct_overbound_bboxes
from face_detector.mtcnn.first_stage import run_first_stage
from face_detector.mtcnn.align_trans import get_reference_facial_points, warp_and_crop_face

class MTCNN():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pnet = PNet().to(self.device)
        self.rnet = RNet().to(self.device)
        self.onet = ONet().to(self.device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square= True)
    
    def get_largest_bbox(self, bboxes, landmarks):
        largest_bbox_idx = 0
        tmp_area = 0        
        for i, b in enumerate(bboxes):
            w = b[2] - b[0]
            h = b[3] - b[1]
            area = w * h #Calculate area for each bounding box
            if(tmp_area < area):
                largest_bbox_idx = i #Get idx for largest bounding box
                tmp_area = area               
        return bboxes[largest_bbox_idx], landmarks[largest_bbox_idx]

    def align(self, img):
        _, landmarks = self.detect_faces(img)
        if landmarks is None or len(landmarks) == 0:
            return None
        facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
        return Image.fromarray(warped_face)
    
    def align_largest_bbox(self, 
                           img, 
                           min_face_size=20.0, 
                           thresholds=[0.70, 0.80, 0.90],
                           nms_thresholds=[0.7, 0.7, 0.7],
                           square_bounding_box=False):
        bboxes, landmarks = self.detect_faces(
            img, 
            min_face_size, 
            thresholds=thresholds,
            nms_thresholds=nms_thresholds, 
            square_bounding_box=square_bounding_box)
        if landmarks is None or len(landmarks) == 0:
            return None, None, None
        
        bbox, landmark = self.get_largest_bbox(bboxes, landmarks)
        facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
        return bbox, Image.fromarray(warped_face), landmark

    def align_multi(self, 
                    img, 
                    limit=None, 
                    min_face_size=20.0,
                    thresholds=[0.70, 0.80, 0.90],
                    nms_thresholds=[0.7, 0.7, 0.7],
                    square_bounding_box=False):
        boxes, landmarks = self.detect_faces(img, 
                                             min_face_size, 
                                             thresholds=thresholds, 
                                             nms_thresholds=nms_thresholds,
                                             square_bounding_box=square_bounding_box)
        if landmarks is None or len(landmarks) == 0:
            return None, None, None
        faces = []
        if len(boxes):
            if limit:
                boxes = boxes[:limit]
                landmarks = landmarks[:limit]

            for landmark in landmarks:
                facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
                # eye1_nose_x = facial5points[2][0]-facial5points[0][0]
                # eye2_nose_x = facial5points[1][0]-facial5points[2][0]                
                # mouth1_nose_x = facial5points[2][0]-facial5points[3][0]
                # mouth2_nose_x = facial5points[4][0]-facial5points[2][0]

                #print("eye1: {}, eye2: {}, nose: {}, mouth1: {}, mouth2: {}".format(
                #    facial5points[0], facial5points[1], facial5points[2], facial5points[3], facial5points[4]))

                # not to include face that face to one side
                # if eye1_nose_x < 0 or eye2_nose_x < 0 or mouth1_nose_x < 0 or mouth2_nose_x < 0:
                #     continue
                
                # ratio_eye = eye1_nose_x / eye2_nose_x
                # if eye1_nose_x > eye2_nose_x:
                #     ratio_eye = eye2_nose_x / eye1_nose_x

                # ratio_mouth = mouth1_nose_x / mouth2_nose_x
                # if mouth1_nose_x > mouth2_nose_x:
                #     ratio_mouth = mouth2_nose_x / mouth1_nose_x

                # if ratio_eye < 0.4 or ratio_mouth < 0.4:
                #     continue

                # print("======================================================")
                #print("eye1_nose_x: {}, eye2_nose_x: {}".format(eye1_nose_x, eye2_nose_x))
                #print("mouth1_nose_x: {}, mouth2_nose_x: {}".format(mouth1_nose_x, mouth2_nose_x))
                warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
                faces.append(Image.fromarray(warped_face))
        return boxes, faces, landmarks

    def detect_faces(self, 
                     image, 
                     min_face_size=20.0,
                     thresholds=[0.70, 0.80, 0.90],
                     nms_thresholds=[0.7, 0.7, 0.7],
                     square_bounding_box=False):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []
        landmarks = None

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(
                    image, 
                    self.pnet, 
                    scale=s, 
                    threshold=thresholds[0], 
                    device=self.device)
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if bounding_boxes:
                bounding_boxes = np.vstack(bounding_boxes)

                keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
                bounding_boxes = bounding_boxes[keep]

                # use offsets predicted by pnet to transform bounding boxes
                bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
                # shape [n_boxes, 5]

                bounding_boxes = convert_to_square(bounding_boxes)
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

                # STAGE 2
                img_boxes = get_image_boxes(bounding_boxes, image, size=24)
                img_boxes = torch.FloatTensor(img_boxes).to(self.device)
                output = self.rnet(img_boxes)
                offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
                probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

                keep = np.where(probs[:, 1] > thresholds[1])[0]
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
                offsets = offsets[keep]

                keep = nms(bounding_boxes, nms_thresholds[1])
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
                bounding_boxes = convert_to_square(bounding_boxes)
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

                # STAGE 3
                img_boxes = get_image_boxes(bounding_boxes, image, size=48)
                if len(img_boxes) == 0:
                    return [], []
                img_boxes = torch.FloatTensor(img_boxes).to(self.device)
                output = self.onet(img_boxes)
                landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
                offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
                probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]
                
                keep = np.where(probs[:, 1] > thresholds[2])[0]
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
                offsets = offsets[keep]
                landmarks = landmarks[keep]

                # compute landmark points
                width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
                height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
                xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
                landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
                landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

                bounding_boxes = calibrate_box(bounding_boxes, offsets)
                keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes = correct_overbound_bboxes(bounding_boxes, image.size[0], image.size[1])
                if (square_bounding_box == True):
                    bounding_boxes = convert_to_square(bounding_boxes)
                landmarks = landmarks[keep]

        return bounding_boxes, landmarks
