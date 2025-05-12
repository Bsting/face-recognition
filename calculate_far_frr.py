import argparse
import time
import csv
import datetime
import os
import logging
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from faceEngine import FaceEngine

def copy_error_file(file_to_copy, error_path, error_folder, error_filename):
    if not os.path.isdir(error_path):
        os.mkdir(error_path)
    error_path = os.path.join(error_path, error_folder)
    if not os.path.isdir(error_path):
        os.mkdir(error_path)
    error_file_path = os.path.join(error_path, error_filename)    
    copyfile(file_to_copy, error_file_path)        

def calculate_score(source_emb, target_emb):
    diff = source_emb.unsqueeze(-1) - target_emb.transpose(1,0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    score, min_idx = torch.min(dist, dim=1)
    return score[0].cpu().numpy()
    
def frr(error_path):
    print('FRR test start: {}'.format (datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))
    begin = time.time()
    
    data_path = 'data/frr_dataset'
    if not os.path.isdir(data_path):
        print('{} does not exist'.format(data_path))
        return
    data_path = Path(data_path)
    error_path = os.path.join(error_path, 'frr')
    if not os.path.isdir(error_path):
        os.mkdir(error_path)
    logging.basicConfig(filename=os.path.join(error_path, 'frr.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    skip_count = 0
    skip_id_count = 0
    output_file = 'frr.csv'
    with open(output_file, mode='w', newline='') as frr_csv:
        csv_writer = csv.writer(frr_csv)
        csv_writer.writerow(['target','subject','extract_feature_time','match_time','score'])
        for img_folder_path in tqdm(data_path.iterdir(), total=len(list(data_path.iterdir()))):
            if img_folder_path.is_file():
                continue

            # set target
            target_file_name = "target.jpg"
            target_file_path = os.path.join(img_folder_path, target_file_name)
            target_emb = None
            try:
                target_img = Image.open(target_file_path)
                target_emb = faceEngine.extract_single(target_img)
                if target_emb == None:
                    skip_id_count += 1
                    copy_error_file(target_file_path, error_path, img_folder_path.name, target_file_name)
                    logging.error('{}: {}'.format(target_file_path, 'Set target failed.'))
                    continue
            except Exception as e:
                skip_id_count += 1
                copy_error_file(target_file_path, error_path, img_folder_path.name, target_file_name)
                logging.error('{}: {}'.format(target_file_path, e))
                continue #if cannot find face/corrupted img, skip
                
            target_name = img_folder_path.name
            for file in img_folder_path.iterdir():
                if not file.is_file() or file.name == 'target.jpg': #skip target.jpg
                    continue
                
                score = 0
                feature_time = 0
                match_time = 0
                try:
                    source_img = Image.open(file)
                    if source_img.size != (112, 112):
                        skip_count += 1
                        copy_error_file(file, error_path, img_folder_path.name, file.name)
                        logging.error('{}: {}'.format(file, 'match failed.'))
                        continue
                    
                    start_time = time.time()
                    source_emb = faceEngine.extract_single(source_img)
                    feature_time = time.time() - start_time
                    
                    if source_emb == None:
                        skip_count += 1
                        copy_error_file(file, error_path, img_folder_path.name, file.name)
                        logging.error('{}: {}'.format(file, 'match failed.'))
                        continue
                    
                    start_time = time.time()
                    score = calculate_score(source_emb, target_emb)
                    match_time = time.time() - start_time

                    csv_writer.writerow([target_name, file.name, '{:4.3f}'.format(feature_time), '{:4.3f}'.format(match_time), '{:8.6f}'.format(score)])
                except Exception as e:
                    skip_count += 1
                    copy_error_file(file, error_path, img_folder_path.name, file.name)
                    logging.error('{}: {}'.format(file, e))
                    continue
    print('{} skipped images: '.format(skip_count))
    print('{} skipped identities: '.format(skip_id_count))
    print('Time taken: {}'.format(time.time() - begin))

def far(error_path):
    print('FAR test start: {}'.format (datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))
    begin = time.time()

    data_path = 'data/far_dataset'
    if not os.path.isdir(data_path):
        print('{} does not exist'.format(data_path))
        return
    data_path = Path(data_path)

    error_path = os.path.join(error_path, 'far')
    if not os.path.isdir(error_path):
        os.mkdir(error_path)
    logging.basicConfig(filename=os.path.join(error_path, 'far.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    
    skip_count = 0
    skip_id_count = 0
    output_file = 'far.csv'
    with open(output_file, mode='w', newline='') as far_csv:
        csv_writer = csv.writer(far_csv)
        csv_writer.writerow(['target','subject','extract_feature_time','match_time','score'])
        for img_folder_path in tqdm(data_path.iterdir(), total=len(list(data_path.iterdir()))):
            if img_folder_path.is_file():
                continue
                
            # set target
            target_file_name = "target.jpg"
            target_file_path = os.path.join(img_folder_path, target_file_name)
            target_emb = None
            target_name = img_folder_path.name
            try:
                target_img = Image.open(target_file_path)
                target_emb = faceEngine.extract_single(target_img)
                if target_emb == None:
                    skip_id_count += 1
                    copy_error_file(target_file_path, error_path, img_folder_path.name, target_file_name)
                    logging.error('{}: {}'.format(target_file_path, 'Set target failed.'))
                    continue
            except Exception as e:
                skip_id_count += 1
                copy_error_file(target_file_path, error_path, img_folder_path.name, target_file_name)
                logging.error('{}: {}'.format(target_file_path, e))
                continue #if cannot find face/corrupted img, skip
            
            for src_img_folder_path in data_path.iterdir():
                if src_img_folder_path.is_file() or src_img_folder_path.name == target_name: #skip own folder
                    continue

                for file in src_img_folder_path.iterdir():
                    if not file.is_file() or file.name[-4:] == '.dat' or file.name == 'target.jpg': #skip .dat/target.jpg
                        continue

                    try:
                        source_img = Image.open(file)
                        if source_img.size != (112, 112):
                            skip_count += 1
                            copy_error_file(file, error_path, src_img_folder_path.name, file.name)
                            logging.error('{}: {}'.format(file, 'match failed.'))
                            continue
                        
                        start_time = time.time()
                        source_emb = faceEngine.extract_single(source_img)
                        feature_time = time.time() - start_time
                        
                        if source_emb == None:
                            skip_count += 1
                            copy_error_file(file, error_path, src_img_folder_path.name, file.name)
                            logging.error('{}: {}'.format(file, 'match failed.'))
                            continue
                        
                        start_time = time.time()
                        score = calculate_score(source_emb, target_emb)
                        match_time = time.time() - start_time
                        
                        source_name = src_img_folder_path.name + '/' + file.name
                        csv_writer.writerow([target_name, source_name, '{:4.3f}'.format(feature_time), '{:4.3f}'.format(match_time), '{:8.6f}'.format(score)])
                    except Exception as e:
                        skip_count += 1
                        copy_error_file(file, error_path, src_img_folder_path.name, file.name)
                        logging.error('{}: {}'.format(file, e))
                        continue #if cannot find face/corrupted img, skip        
        print('{} skipped images.'.format(skip_count))
        print('{} skipped identities.'.format(skip_id_count))
        print('Time taken: {}'.format(time.time() - begin))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for calculate FRR/FAR')
    parser.add_argument("-far", "--far", help="calculate FAR",action="store_true")
    args = parser.parse_args()
    
    faceEngine = FaceEngine()
    faceEngine.load_state('pretrained_model/model.pth')
    print(faceEngine.conf.device)
    torch.set_num_threads(4)
    error_path = 'data/error'
    if not os.path.isdir(error_path):
        os.mkdir(error_path)
    
    if not args.far: #calculate FRR
        frr(error_path)
    else: #calculate FAR
        far(error_path)
