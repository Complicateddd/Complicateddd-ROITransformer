def draw_poly_detections(img_path, anno_path):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb
    import cv2
    import random
    import mmcv
    class_names=('1','2','3','4','5')
    # assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img_path)
    color_white = (255, 255, 255)
    f=open(anno_path,'r')
    rrbox=[]
    for ff in f.readlines():
        rrbox.append(ff.strip().split(" "))
    # print(rrbox)

    
    for j, rbox in enumerate(rrbox):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    #     try:
    #         dets = detections[j]
        # dets=
    #     except:
    #         pdb.set_trace()
    #     for det in dets:
    #         # print(det)
        bbox = rbox[1:9]
        clas=rbox[0]

    #         score = det[-1]
        bbox = list(map(int, bbox))

        cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
        for i in range(3):
            cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2)
        cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
        cv2.putText(img, '%s %.3f' % (clas, 1), (bbox[0], bbox[1] + 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img


if __name__ == '__main__':
    import cv2
    import os
    file_name_list=os.listdir("/media/ubuntu/data/huojianjun/科目四初赛第一阶段/train_all_all/images")
    for file in file_name_list:
        # print(file[:-4])
        path="/media/ubuntu/data/huojianjun/科目四初赛第一阶段/train_all_all/images/"+file[:-4]+".tif"
        path2="/media/ubuntu/data/huojianjun/科目四初赛第一阶段/train_all_all/labelTxt/"+file[:-4]+".txt"

        # print(file_name_list)


        img=draw_poly_detections(path,path2)
        path3="/media/ubuntu/data/huojianjun/科目四初赛第一阶段/groundtruth/"+file[:-4]+".tif"
        cv2.imwrite(path3, img)
