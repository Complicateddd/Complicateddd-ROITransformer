import os
import random
import shutil

file=os.listdir("/media/ubuntu/data/huojianjun/科目四热身赛数据/labelTxt")
tv=int(len(file)*0.8)

list_one=list(range(1,len(file)+1))
trainval=random.sample(list_one,tv)

for i in list_one:
	if i in trainval:
		shutil.copy(os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/images/{}.tif'.format(i)),
			os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/trainval/images/{}.tif'.format(i)))

		shutil.copy(os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/labelTxt/{}.txt'.format(i)),
			os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/trainval/labelTxt/{}.txt'.format(i)))
	else:
		shutil.copy(os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/images/{}.tif'.format(i)),
			os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/test/images/{}.tif'.format(i)))

		shutil.copy(os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/labelTxt/{}.txt'.format(i)),
			os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/test/labelTxt/{}.txt'.format(i)))
# print(list_one)



# import os
# import shutil
# file=open("/media/ubuntu/新加卷/xiangmu/dataset/ImageSets/Main/test.txt",'r')
# list_=[]
# for line in file.readlines():
# 	list_.append(line.strip()+'.jpg')
# 	print(line)
# print(list_)
# img=os.listdir("/media/ubuntu/新加卷/xiangmu/dataset/JPEGImages")
# print(len(img))
# for i in img:
# 	if i in list_:
# 		shutil.copy(os.path.join("/media/ubuntu/新加卷/xiangmu/dataset/JPEGImages",i),
# 			os.path.join("/media/ubuntu/新加卷/xiangmu/sample",i))
# file.close()