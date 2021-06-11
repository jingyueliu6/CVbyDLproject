from image_process import load_data
from network import FCN
from tqdm import tqdm
import sys
import os
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

path_images = './images/'
path_annotations = './annotations/'
output_folder = './model/'

files = []
airports = sorted(os.listdir(path_annotations)) # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
                                                # sorted() 函数对所有可迭代的对象进行排序操作。
for airport in airports:
	fs = [f for f in sorted(os.listdir(os.path.join(path_annotations, airport))) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d\.txt', f)]
	files += [(os.path.join(path_images, airport, f.replace('.txt', '_image.png')), os.path.join(path_annotations, airport, f)) for f in fs]

def train(det_model, optimizer, imgs, labels, batch_size, iterations, epoch):
	with tqdm(total=iterations, file=sys.stdout) as pbar:
		pbar.set_description('Epoch #{} of training)'.format(epoch+1))

		det_model.train()
		for i in range(iterations):
			# randomly select batch_size images from the training set
			batch = np.random.permutation(len(imgs))[:batch_size]
			np_batch_data = imgs.take(batch, axis=0)
			# create torch tensors for images and labels of the current batch
			batch_data = torch.from_numpy(np.transpose(np_batch_data, (0, 3, 1, 2))).cuda()
			batch_target = torch.tensor(labels.take(batch, axis=0), dtype=torch.long).float().cuda()

			# run one optimization step
			optimizer.zero_grad()
			batch_pred = det_model(batch_data).squeeze()
			loss = F.binary_cross_entropy(batch_pred, batch_target)
			loss.backward()
			optimizer.step()
			y = batch_target.cpu().detach().numpy()
			pre = batch_pred.cpu().detach().numpy()
			accuracy = (1 - np.sum(np.absolute(y - pre)) / y.shape[0]) *100

			# check performance every 100 iterations
			if i%100 == 99:
				pbar.set_description('Epoch #{} of training (loss {:.2f}%, accuracy {:.2f}%)'.format(epoch + 1, loss.item(), accuracy))
			pbar.update(1)
	return loss.item(), accuracy

train_pos, train_neg = load_data(files)
train_imgs = np.concatenate((train_pos,train_neg), axis=0)
train_labels = np.asarray([1]*len(train_pos) + [0]*len(train_neg), dtype=np.int32)
detect_model = FCN().cuda()
# detect_model = FCN()
# num_iter = 3000
num_iter = 30
batch_size = 256
optimizer = optim.Adam(detect_model.parameters(), lr=0.0001)
max_epochs = 20
# for epoch in range(1, max_epochs):
#     with tqdm(total=num_iter, file=sys.stdout) as pbar:
#         pbar.set_description('Epoch #{} of training'.format(epoch + 1))
#
#         detect_model.train() # 必备，将模型设置为训练模式
#         for i in range(num_iter):
#             # randomly select batch_size images from the training set
#             batch = np.random.permutation(len(train_imgs))[:batch_size]
#             np_batch_data = train_imgs.take(batch, axis=0)
#
#             # create torch tensors for images and labels of the current batch
#             batch_data = torch.from_numpy(np.transpose(np_batch_data, (0, 3, 1, 2))).cuda()
#             batch_target = torch.tensor(train_labels.take(batch, axis=0), dtype=torch.long).float().cuda()
#             # batch_data = torch.from_numpy(np.transpose(np_batch_data, (0, 3, 1, 2)))
#             # batch_target = torch.tensor(train_labels.take(batch, axis=0), dtype=torch.long).float()
#
#             # run optimization step
#             optimizer.zero_grad() # 清除所有优化的梯度
#             batch_pred = detect_model(batch_data).squeeze() # 喂入数据并前向传播获取输出
#             loss = F.binary_cross_entropy(batch_pred, batch_target) # 调用损失函数计算损失
#             loss.backward() # 反向传播
#             optimizer.step() # 更新参数
#             y = batch_target.numpy()
#             pre = batch_pred.numpy()
#             accuracy = np.sum(np.absolute(y - pre)) / y.shape[0]
#             loss_list.append(loss.item())
#             accuracy_list.append(accuracy)
#             # check performance every 100 iterations
#             if i % 100 == 0:
#                 pbar.set_description('Epoch #{} of training (loss {:.2f}%, accuracy {:.2f}%)'.format(epoch + 1, loss.item(), accuracy))
#     loss_list.append(loss.item())
#     accuracy_list.append(accuracy)
loss_list = []
accuracy_list = []
for epoch in range(0, max_epochs):
    loss, acc = train(detect_model, optimizer, train_imgs, train_labels, batch_size, num_iter, epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
x1 = range(0, max_epochs)
x2 = range(0, max_epochs)
y1 = accuracy_list
y2 = loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1,'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.savefig("accuracy_loss.png")
plt.show()

torch.save(detect_model.state_dict(), os.path.join(output_folder, 'flying.pytorch'))
