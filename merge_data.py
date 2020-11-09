import os
import shutil
file=os.listdir("/media/ubuntu/data/huojianjun/科目四热身赛数据/images")
for fi in file:
	name_ori_id=int(fi[:-4])
	# print(name_ori_id)
	trains_id=name_ori_id+2008
	# print(trains_id)
	shutil.copy(os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/images/{}.tif'.format(name_ori_id)),
			os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/reshen/images/{}.tif'.format(trains_id)))
	shutil.copy(os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/labelTxt/{}.txt'.format(name_ori_id)),
			os.path.join('/media/ubuntu/data/huojianjun/科目四热身赛数据/reshen/labelTxt/{}.txt'.format(trains_id)))