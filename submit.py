from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
import numpy as np
from tqdm import tqdm
import DOTA_devkit.polyiou as polyiou
import math
import pdb

def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class DetectorModel():
    def __init__(self,
                 config_file,
                 checkpoint_file):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        # self.classnames = self.dataset.CLASSES
        self.classnames = ('1', '2', '3', '4', '5')

        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def inference_single(self, imagname):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        # slide_h, slide_w = slide_size
        # hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]
        # print(self.classnames)

        chip_detections = inference_detector(self.model, img)
        # nms
        for i in range(5):
            keep = py_cpu_nms_poly_fast_np(chip_detections[i], 0.1)
            chip_detections[i] = chip_detections[i][keep]
        return chip_detections

    def inference_single_vis(self, srcpath, dstpath):
        detections = self.inference_single(srcpath)
        print(detections)
        img = draw_poly_detections(srcpath, detections, self.classnames, scale=1, threshold=0.3)
        cv2.imwrite(dstpath, img)

if __name__ == '__main__':
    import tqdm
    roitransformer = DetectorModel(r'configs/Huojianjun/faster_rcnn_RoITrans_r101x_fpn_1x_anchors_augs_augfpn.py',
                  r'work_dirs/faster_rcnn_RoITrans_r101_all_aug_rote_1333_crop_rote/epoch_278.pth')

    # roitransformer.inference_single_vis(r'demo/48.tif',
    #                                    r'demo/48_out.tif',
    #                                     (1024, 1024),
    #                                    (1024, 1024))

    threshold=0.0001
    class_names=('1', '2', '3', '4', '5')
    import os
    path="/media/ubuntu/data/huojianjun/科目四/科目四/test2"
    file_img_name=os.listdir(path)

    result_file=open("./科目四_莘莘学子.txt",'w')

    # print(file_img_name)
    count=0
    def filer(x):
        x=int(x)
        if x>1024:
            return 1024
        if x<0:
            return 0
        else:
            return x

    for name in tqdm.tqdm(file_img_name):
        # count+=1
        path_img=os.path.join(path,name)
        detection_result=roitransformer.inference_single(path_img)
        for j, name_cls in enumerate(class_names):
            dets = detection_result[j]
            for det in dets:
                bbox = det[:8]
                score = round(det[-1],2)
                if score < threshold:
                    continue
                bbox = list(map(filer, bbox))
                # print(bbox)
                # print(score)
                # print(name_cls)
                result_file.writelines(name+" "+str(name_cls)+" "+str(score)+" "
                +str(bbox[0])
                +" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])
                    +" "+str(bbox[4])+" "+str(bbox[5])+" "+str(bbox[6])
                        +" "+str(bbox[7]))
                result_file.writelines("\n")
                count+=1
                # if name=="3.tif":
                #     print(count)
        # if count==3:

        #     break

        # print(path_img)

