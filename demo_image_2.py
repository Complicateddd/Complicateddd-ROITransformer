from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections,inference_detector_2
from mmdet.apis import draw_poly_detections_2,init_detector_2
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
        self.cfg_2=Config.fromfile(self.config_file)

        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        # self.classnames = self.dataset.CLASSES
        self.classnames = ('1', '2', '3', '4', '5')

        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

        self.cfg_2.data['test']['img_scale']=(1666,1666)
        self.cfg_2.test_cfg['rcnn']['score_thr']=0.25

        self.model_2=init_detector_2(self.cfg_2, checkpoint_file, device='cuda:0')


# config.test_cfg
        # print(self.cfg.data['test']['img_scale'])
    def inference_single(self, imagname, slide_size, chip_size):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        slide_h, slide_w = slide_size
        hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = np.zeros((0, 9))
        # print(self.classnames)

        chip_detections = inference_detector(self.model, img)

        chip_detections_2=inference_detector(self.model_2, img)
        # for i in range(5):

        #     print('result: ', chip_detections[i])
        # for i in tqdm(range(int(width / slide_w + 1))):
        #     for j in range(int(height / slide_h) + 1):
        #         subimg = np.zeros((hn, wn, channel))
        #         # print('i: ', i, 'j: ', j)
        #         chip = img[j*slide_h:j*slide_h + hn, i*slide_w:i*slide_w + wn, :3]
        #         subimg[:chip.shape[0], :chip.shape[1], :] = chip

        #         chip_detections = inference_detector(self.model, subimg)

        #         print('result: ', chip_detections)
        #         for cls_id, name in enumerate(self.classnames):
        #             # chip_detections[cls_id][:, :8][:, ::2] = chip_detections[cls_id][:, :8][:, ::2] + i * slide_w
        #             # chip_detections[cls_id][:, :8][:, 1::2] = chip_detections[cls_id][:, :8][:, 1::2] + j * slide_h
        #             # import pdb;pdb.set_trace()
        #             # try:
        #             total_detections[cls_id] = chip_detections[cls_id]
                    # except:
                    #     import pdb; pdb.set_trace()
        # nms
        # total_detections=chip_detections
        # print(chip_detections.shape)
        # for i in range(5):
        #     # print(len(chip_detections[i]))
        #     if len(chip_detections[i]):
        #         # print(chip_detections[i].shape)
        #         # print(total_detections)
        #         total_detections=np.concatenate((total_detections,chip_detections[i]))
        # # print(total_detections[1:].shape)
        # total_detections_=total_detections[1:]
        # print(chip_detections)
        
        # totol_class=np.zeros((0,1))
        # for i in range(5):
        #     total_detections=np.concatenate((total_detections,chip_detections[i]))
        #     total_detections=np.concatenate((total_detections,chip_detections_2[i]))

        #     # print(chip_detections[i].shape[0])
        #     temp_class=np.ones((chip_detections[i].shape[0],1))*i
        #     totol_class=np.concatenate((totol_class,temp_class))

        #     temp_class=np.ones((chip_detections_2[i].shape[0],1))*i
        #     totol_class=np.concatenate((totol_class,temp_class))

        # keep = py_cpu_nms_poly_fast_np(total_detections, 0.1)
        # totol_class=totol_class[keep]
        # total_detections=total_detections[keep]

        # print(total_detections.shape)
        for i in range(5):
            # print(chip_detections[i].shape)
            chip_detections[i]=np.concatenate((chip_detections[i],chip_detections_2[i]))
            keep = py_cpu_nms_poly_fast_np(chip_detections[i], 0.1)

            chip_detections[i] = chip_detections[i][keep]

        return chip_detections
        # keep=py_cpu_nms_poly_fast_np(total_detections_, 0.1)
        # total_detections_=total_detections_[keep]
        # # print(total_detections_)
        # return total_detections,totol_class

# 
    def inference_single_vis(self, srcpath, dstpath, slide_size, chip_size):
        # detections,totol_class = self.inference_single(srcpath, slide_size, chip_size)
        detections= self.inference_single(srcpath, slide_size, chip_size)

        # print(detections)
        # img = draw_poly_detections_2(srcpath, detections, totol_class,self.classnames, scale=1, threshold=0.05)
        img = draw_poly_detections(srcpath, detections,self.classnames, scale=1, threshold=0.05)

        cv2.imwrite(dstpath, img)

if __name__ == '__main__':
    import tqdm
    roitransformer = DetectorModel(r'work_dirs/faster_rcnn_RoITrans_r101_fpn_1x_all_aug/faster_rcnn_RoITrans_r101x_fpn_1x_anchors_augs_augfpn.py',
                  r'work_dirs/faster_rcnn_RoITrans_r101_fpn_1x_all_aug/epoch_140.pth')

    # roitransformer.inference_single_vis(r'parse/244.tif',
    #                                    r'parse/244_out.tif',
    #                                     (1024, 1024),
    #                                    (1024, 1024))

    import os
    path="/media/ubuntu/data/huojianjun/科目四初赛第一阶段/test1"
    # path="/media/ubuntu/data/huojianjun/科目四热身赛数据/trainval/images"
    file_img_name=os.listdir(path)
    for name in tqdm.tqdm(file_img_name):
        path_img=os.path.join(path,name)
        roitransformer.inference_single_vis(path_img,r'demo_re/{}'.format(name),(1024, 1024),(1024, 1024))
        # break



