from __future__ import print_function
import torch
import cv2
import torch.backends.cudnn as cudnn
import numpy as np
from face_detector.faceboxes.layers.functions.prior_box import PriorBox
from face_detector.faceboxes.utils.nms.py_cpu_nms import py_cpu_nms, nms
from face_detector.faceboxes.models.faceboxes import FaceBoxes
from face_detector.faceboxes.utils.box_utils import decode

cfg = {
    'name': 'FaceBoxes',
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'min_dim': 1024,
    'steps': [32, 64, 128],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'aspect_ratios': [[1], [1], [1]],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}

class Faceboxes():
    def __init__(self):
         # net and model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        self.net = load_model(self.net, 'pretrained_model/faceboxes.pth', True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True        
        self.net = self.net.to(self.device)

    def detect_faces(self, image, resize=2, confidence_threshold=0.75, nms_threshold=0.3, top_k=5000, keep_top_k=750):
        with torch.no_grad():
            image = np.float32(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            im_height, im_width, _ = image.shape
            #print(image.shape)
            if resize != 1:
                image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            image -= (104, 117, 123)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0)
            image = image.to(self.device)
            scale = scale.to(self.device) / resize

            out = self.net(image)  # forward pass
            priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
            priors = priorbox.forward()
            priors = priors.to(self.device)
            loc, conf, _ = out
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, 1]

            # ignore low scores
            #print('decode boxes: {0}'.format(boxes.shape))
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]
            #print('after confidence_threshold boxes: {0}'.format(boxes.shape))
            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            scores = scores[order]
            #print('after top_k boxes: {0}'.format(boxes.shape))
            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            #keep = py_cpu_nms(dets, args.nms_threshold)
            keep = nms(dets, nms_threshold)
            dets = dets[keep, :]
            #print('after nms_threshold boxes: {0}'.format(dets.shape))
            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            #print('after keep_top_k boxes: {0}'.format(dets.shape))
            #bounding_boxes = []
            bounding_boxes = np.zeros(shape=(dets.shape[0], 4))
            for k in range(dets.shape[0]):
                bounding_boxes[k, 0] = dets[k, 0]
                bounding_boxes[k, 1] = dets[k, 1]
                bounding_boxes[k, 2] = dets[k, 2]
                bounding_boxes[k, 3] = dets[k, 3]               
                
        return bounding_boxes

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #print('Missing keys:{}'.format(len(missing_keys)))
    #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    #print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model