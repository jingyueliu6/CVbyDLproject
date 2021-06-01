import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import json
import sys
import cv2
import os
import re

from network import FCN
from network import NMS

path_images = './images/'
output_folder = './logs/'

def detect_image(det_model, nms_model, img):
	block = 512
	radius = 25
	size = 51

	# compute mask of detection for satellite image in chunks of [block x block] pixels

	full_mask = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
	with torch.no_grad():
		for y in range(0, img.shape[0], block-2*radius):
			if y+size > img.shape[0]:
				break
			for x in range(0, img.shape[1], block-2*radius):
				if x+size > img.shape[1]:
					break
				if np.sum(img[y:y+block,x:x+block]) == 0:
					continue
				img_crop = torch.from_numpy(np.transpose(img[y:y+block,x:x+block], (2,0,1))).float().unsqueeze(0).cuda()
				mask_crop = det_model(img_crop)
				full_mask[y+radius:min(y+block-radius,img.shape[0]-radius), x+radius:min(x+block-radius,img.shape[1]-radius)] = mask_crop[0, :, :, :].cpu().numpy().transpose((1,2,0))

	det_mask = torch.from_numpy(np.transpose(full_mask, (2,0,1))).float().unsqueeze(0).cuda()
	dets = nms_model(det_mask).cpu().numpy()[:,2:].tolist()

	return dets

det_model = FCN().cuda()
det_model.load_state_dict(torch.load('./model/flying.pytorch'))
det_model.eval()

nms_model = NMS().cuda()
nms_model.eval()

airports = sorted(os.listdir(path_images))
for airport in airports:
	files = [f for f in sorted(os.listdir(os.path.join(path_images, airport))) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d_image\.png', f)]

	log = {}
	for f in files:
		print(airport, f)

		# load satellite image
		img = cv2.imread(os.path.join(path_images, airport, f), cv2.IMREAD_COLOR)[:, :, ::-1]/255.0

		# detect airplanes
		detects = detect_image(det_model, nms_model, img)

		block_height = img.shape[0]//7 # 7791//7 = 1113
		block_width = img.shape[1]//7 # 6965//7 = 995
		## 49 blocks for one image
		timestamp = f.split('_')[0]

		valid = []
		flag = False

		for i in range(0,img.shape[0],block_height):
			for j in range(0,img.shape[1],block_width):
				if np.sum(img[i:i+block_height,j:j+block_width]) > 0:
					valid.append(0) # valid place
				else:
					valid.append(None) # cloud and blank ground

		for c in detects:
			pos = (c[0]//block_height)*7 + c[1]//block_width
			if valid[pos] is not None:
				valid[pos] += 1

		for i in range(49):
			if valid[i] is not None and valid[i] > 5:
				valid[i] = None

		if valid.count(None) < len(valid):
			log[timestamp] = valid

	json.dump(log, open(os.path.join(output_folder, airport+'.log'), 'w'))