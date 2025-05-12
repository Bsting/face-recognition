from mtcnn import MTCNN
from faceEngine import FaceEngine
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os
import time
import torch

def prepare_facebank(faceEngine, mtcnn, log_file, dummy=False):
    embeddings = []
    names = []
    dataPath = os.path.join(faceEngine.conf.facebank_path, 'dataset')
    for file in Path(dataPath).iterdir():
        if not file.is_file():
            continue
        
        facial_features = extract_face_features(faceEngine, mtcnn, file, log_file)
        if facial_features is None:
            continue
        
        if (dummy == True):
            if 'id' not in file.name:
                embedding = torch.cat(facial_features).mean(0, keepdim=True)
                embeddings.append(embedding)
                names.append(file.stem)
            else:
                for i in range(100000):
                    embedding = torch.cat(facial_features).mean(0, keepdim=True)
                    embeddings.append(embedding)
                    names.append('{0}_{1}'.format(file.stem, i))
        else:
            embedding = torch.cat(facial_features).mean(0, keepdim=True)
            embeddings.append(embedding)
            names.append(file.stem)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, os.path.join(faceEngine.conf.facebank_path, 'facebank.pth'))
    np.save(os.path.join(faceEngine.conf.facebank_path, 'names'), names)
    return embeddings, names

def update_facebank(faceEngine, mtcnn, log_file, dummy=False):
    embeddings = torch.load(os.path.join(faceEngine.conf.facebank_path, "facebank.pth"))
    names = np.load(os.path.join(faceEngine.conf.facebank_path, "names.npy"))
    print(embeddings.shape)
    print(names.shape)
    dataPath = os.path.join(faceEngine.conf.facebank_path, 'update_dataset')
    log_file = "update_facebank.log"
    for file in Path(dataPath).iterdir():
        if not file.is_file():
            continue
        
        facial_features = extract_face_features(faceEngine, mtcnn, file, log_file)
        if facial_features is None:
            continue
        
        if (dummy == True):
            if 'id' not in file.name:
                embedding = torch.cat(facial_features).mean(0, keepdim=True)
                embeddings = torch.cat((embeddings, embedding), dim=0)
                names = np.append(names, file.stem)
            else:
                for i in range(100000):
                    embedding = torch.cat(facial_features).mean(0, keepdim=True)
                    embeddings = torch.cat((embeddings, embedding), dim=0)
                    names = np.append(names,'{0}_{1}'.format(file.stem, i))
        else:
            embedding = torch.cat(facial_features).mean(0, keepdim=True)
            embeddings = torch.cat((embeddings, embedding), dim=0)
            names = np.append(names, file.stem)

    torch.save(embeddings, os.path.join(faceEngine.conf.facebank_path, 'facebank.pth'))
    np.save(os.path.join(faceEngine.conf.facebank_path, 'names'), names)
    return embeddings, names

def extract_face_features(faceEngine, mtcnn, file, log_file):
    reject_path = os.path.join(faceEngine.conf.facebank_path, 'reject')
    face_path = os.path.join(faceEngine.conf.facebank_path, 'face')
    try:
        img = Image.open(file)
    except:
        write_log('Failed to open image file, skip {0}'.format(file.name), log_file)
        img.save(os.path.join(reject_path, f'{file.stem}_failed_open.jpg'))
        return None
    
    _, face, landmark = mtcnn.align_largest_bbox(
            img,
            65, 
            thresholds=[0.80, 0.95, 0.99])
            
    if face is None:
        write_log('No face detected, skip {0}'.format(file.name), log_file)
        img.save(os.path.join(reject_path, f'{file.stem}_failed_detect_face.jpg'))
        return None

    facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
    eye1_nose_x = facial5points[2][0]-facial5points[0][0] 
    eye2_nose_x = facial5points[1][0]-facial5points[2][0]  
    mouth1_nose_x = facial5points[2][0]-facial5points[3][0]
    mouth2_nose_x = facial5points[4][0]-facial5points[2][0]

    #print("eye1: {}, eye2: {}, nose: {}, mouth1: {}, mouth2: {}".format(
    #   facial5points[0], facial5points[1], facial5points[2], facial5points[3], facial5points[4]))

    # Not to include face that face to one side
    if eye1_nose_x < 5 or eye2_nose_x < 5 or mouth1_nose_x < 5 or mouth2_nose_x < 5:
        write_log(f'Side face detected, skip {file.name}', log_file)
        write_log(f'eye1_nose_x: {eye1_nose_x}, eye2_nose_x: {eye2_nose_x}', log_file)
        write_log(f'mouth1_nose_x: {mouth1_nose_x}, mouth2_nose_x: {mouth2_nose_x}', log_file)
        img.save(os.path.join(reject_path, file.name))
        face.save(os.path.join(reject_path, f'{file.stem}_side_face.jpg'))
        return None
    
    eye_distance = facial5points[1][0] - facial5points[0][0]
    if eye_distance < 15:
        write_log(f'Side face detected, skip {file.name}', log_file)
        write_log(f'eye1_nose_x: {eye1_nose_x}, eye2_nose_x: {eye2_nose_x}', log_file)
        img.save(os.path.join(reject_path, file.name))
        face.save(os.path.join(reject_path, f'{file.stem}_side_face.jpg'))
        return None
    
    cv_face = np.array(face)
    blur_score = get_blur_score(cv_face[:, :, ::-1].copy())
    if blur_score < 30:
        write_log(f'Face is blur, blur score {blur_score}', log_file)
        img.save(os.path.join(reject_path, file.name))
        face.save(os.path.join(reject_path, f'{file.stem}_blur_{blur_score}.jpg'))
        return None
    
    embeddings = []
    with torch.no_grad():
        embeddings.append(faceEngine.model(faceEngine.conf.transform(face).to(faceEngine.conf.device).unsqueeze(0)))
    if len(embeddings) == 0:
        write_log(f'Failed to extract embedding {blur_score}', log_file)
        img.save(os.path.join(reject_path, file.name))
        face.save(os.path.join(reject_path, f'{file.stem}_failed_extract.jpg'))
        return None
    face.save(os.path.join(face_path, file.name))
    return embeddings

def write_log(message, log_file='facebank.log'):
    print(message)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
    with open(log_file, 'a') as file:  # Open file in append mode
        file.write(f'{timestamp} - {message}\n')

def get_blur_score(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian	
	# remove noise by blurring with a Gaussian filter
	imageBlur = cv2.GaussianBlur(image, (3,3), 0)
	imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
	return cv2.Laplacian(imageGray, cv2.CV_64F).var()

if __name__ == '__main__':
    log_file='facebank.log'
    update = False
    if update:
        log_file = "update_facebank.log"
    write_log("Starting...", log_file)

    mtcnn = MTCNN()
    write_log('Face detector loaded.', log_file)

    faceEngine = FaceEngine()
    #learner.load_state('pretrained_model/model_ir_se50.pth')
    faceEngine.load_state('pretrained_model/model.pth')
    faceEngine.model.eval()
    write_log('Face engine loaded.', log_file)

    write_log('Preparing Facebank...', log_file)
    if update:
        targets, names = update_facebank(faceEngine, mtcnn, log_file, False) 
    else:
        targets, names = prepare_facebank(faceEngine, mtcnn, log_file, False) 
    write_log(f'Facebank created/updated, size {len(targets)}.', log_file)