from tqdm import tqdm
import numpy as np
import random
import sys
import cv2
import os
import re

# path_images = './images/'
# path_annotations = './annotations/'
#
# files = []
# airports = sorted(os.listdir(path_annotations)) # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
#                                                 # sorted() 函数对所有可迭代的对象进行排序操作。
# print(airports) #输出： ['BRU', 'DUB']
# for airport in airports:
# 	fs = [f for f in sorted(os.listdir(os.path.join(path_annotations, airport))) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d\.txt', f)]
# 	files += [(os.path.join(path_images, airport, f.replace('.txt', '_image.png')), os.path.join(path_annotations, airport, f)) for f in fs]
#
# print(files[0]) #输出： ('./images/DUB\\2020-01-03-11-36-21_image.png', './annotations/DUB\\2020-01-03-11-36-21.txt')
# # 显示原始图片
# img = cv2.imread(files[1][0], cv2.IMREAD_COLOR)
# print(img.shape[0]) #7791
# print(img.shape[1]) #6965
# cv2.imshow('image_org',img)
# cv2.waitKey(0)
# img = cv2.imread(files[1][0], cv2.IMREAD_COLOR)/255.0
# cv2.imshow('image_mod1',img)
# cv2.waitKey(0)
# img = cv2.imread(files[1][0], cv2.IMREAD_COLOR)[:, :, ::-1]/255.0 #原本的bgr排列方式经过倒序就变成了rgb的通道排列方式。opencv默认是以BGR通道顺序打开的和显示的
# cv2.imshow('image_mod',img)
# cv2.waitKey(0)

# positives = []
# negatives = []
# ann_list = []
# with open(files[1][1], 'r') as f:
# 	for line in f:
# 		row, col = [int(x) for x in line.split()]
# 		ann_list.append((row,col))
# print(ann_list) # [(4162, 5408), (4021, 3954), (5042, 2687)]
# size = 25
# step = 3
# for cc in ann_list:
# 	for x in range(-1, 2):
# 		for y in range(-1, 2):
# 			# positive samples
# 			c = (cc[0]+y*step, cc[1]+x*step)
# 			if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
# 				positives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())
# 			# negative samples
# 			if x != 0 or y != 0:
# 				c = (cc[0]+y*size, cc[1]+x*size)
# 				if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
# 					negatives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())
# print(len(positives)) #27
# print(len(negatives)) #24
# cv2.imshow('target_place',img[ann_list[0][0]+-size:ann_list[0][0]+size+1,ann_list[0][1]-size:ann_list[0][1]+size+1])
# cv2.waitKey(0)
# cv2.imshow('negative_example',img[ann_list[0][0]+25-size:ann_list[0][0]+25+size+1,ann_list[0][1]-size:ann_list[0][1]+size+1])
# cv2.waitKey(0)

def load_data(files):
	positives = []
	negatives = []
	with tqdm(total=len(files), file=sys.stdout) as pbar:
		pbar.set_description('Parsing training data')
		for img_name, ann_name in files:
			# load airplane annotations
			ann_list = []
			with open(ann_name, 'r') as f:
				for line in f:
					row, col = [int(x) for x in line.split()]
					ann_list.append((row,col))

			# load satellite image
			img = cv2.imread(img_name, cv2.IMREAD_COLOR)[:, :, ::-1]/255.0

			# crop samples from input image
			size = 25
			step = 3
			for cc in ann_list:
				for x in range(-1, 2):
					for y in range(-1, 2):
						# positive samples
						c = (cc[0]+y*step, cc[1]+x*step)
						if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
							positives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())
						# negative samples
						if x != 0 or y != 0:
							c = (cc[0]+y*size, cc[1]+x*size)
							if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
								negatives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())

			# extra negative samples sampled randomly over the entire image
			while len(negatives) < 2*len(positives):
				c = (np.random.randint(img.shape[0]), np.random.randint(img.shape[1]))
				if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
					flag = True
					for cc in ann_list:
						if abs(cc[0]-c[0]) <= size or abs(cc[1]-c[1]) <= size:
							flag = False
							break
					# discard if sampled point is too close to an annotated point or if it falls in a blank image region
					if flag and np.sum(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1]) > 0:
						negatives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())

			pbar.update(1)

	# keep a 1:2 ratio limit between positive and negtive samples
	if len(negatives) > 2*len(positives):
		negatives = random.sample(negatives, 2*len(positives))

	return np.asarray(positives, dtype=np.float32), np.asarray(negatives, dtype=np.float32)

