import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer as DC
from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector
# from .rotate_aug import RotateAugmentation
from .rotate_aug import RotateTestAugmentation
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        print("inference_detector")
        return _inference_generator(model, imgs, img_transform, device)


# def prepare_aug_data(img, img_transform, cfg, device):
#     ori_shape = img.shape
#     img, img_shape, pad_shape, scale_factor = img_transform(
#         img,
#         scale=cfg.data.test.img_scale,
#         keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
#     img = to_tensor(img).to(device).unsqueeze(0)

#     print(cfg.data.test.img_scale)

#     img_meta = [
#         dict(
#             ori_shape=ori_shape,
#             img_shape=img_shape,
#             pad_shape=pad_shape,
#             scale_factor=scale_factor,
#             flip=False)
#     ]
#     return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = prepare_aug_data(img, img_transform, model.cfg, device)

    model = MMDataParallel(model, device_ids=[0])

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


# TODO: merge this method with the one in BaseDetector
def show_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)

def draw_poly_detections(img, detections, class_names, scale, threshold=0.2):
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
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for j, name in enumerate(class_names):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        for det in dets:
            # print(det)
            bbox = det[:8] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = list(map(int, bbox))

            cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
            cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        # for index in range(dets.shape[0]):
            # print(dets)
        # bbox = dets[index][:8] * scale
        # score = dets[index][-1]
        # if score < threshold:
        #     continue
        # bbox = list(map(int, bbox))

        # cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
        # for i in range(3):
        #     cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2)
        # cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
        # cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
        #             color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img





#Note muti 

def prepare_aug_data(img, img_transform, cfg, device):
        """Prepare an image for testing (multi-scale and flipping)"""
        # img_info = self.img_infos[idx]
        # img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # if self.proposals is not None:
        #     proposal = self.proposals[idx][:self.num_max_proposals]
        #     if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
        #         raise AssertionError(
        #             'proposals should have shapes (n, 4) or (n, 5), '
        #             'but found {}'.format(proposal.shape))
        # else:
        rotate_test_aug = RotateTestAugmentation()
        proposal = None
        # TODO: make the flip and rotate at the same time
        # TODO: when implement the img rotation, we do not consider the proposals, add it in future
        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = img_transform(
                img, scale, flip, keep_ratio=True)
            _img = to_tensor(_img).to(device).unsqueeze(0)
            _img_meta = dict(
                ori_shape=(1024,1024, 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                angle=0)
            # if proposal is not None:
            #     if proposal.shape[1] == 5:
            #         score = proposal[:, 4, None]
            #         proposal = proposal[:, :4]
            #     else:
            #         score = None
            #     _proposal = self.bbox_transform(proposal, img_shape,
            #                                     scale_factor, flip)
            #     _proposal = np.hstack(
            #         [_proposal, score]) if score is not None else _proposal
            #     _proposal = to_tensor(_proposal)
            # else:
            _proposal = None
            return _img, _img_meta, _proposal

        def prepare_rotation_single(img, scale, flip, angle):
            _img, img_shape, pad_shape, scale_factor = rotate_test_aug(
                img, angle=angle)
            _img, img_shape, pad_shape, scale_factor = img_transform(
                _img, scale, flip, keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
            _img = to_tensor(_img).to(device).unsqueeze(0)
            # if self.rotate_test_aug is not None:
            _img_meta = dict(
                ori_shape=(1024,1024, 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                angle=angle
            )
            return _img, _img_meta

        imgs = []
        img_metas = []
        # proposals = []
        for scale in cfg.data.test.img_scale:
            # print("scale",scale)
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            # img_metas.append(DC(_img_meta, cpu_only=True))
            img_metas.append(_img_meta)
            # proposals.append(_proposal)
            if cfg.data.test.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                # proposals.append(_proposal)
        if cfg.data.test.rotate_test_aug is not None :
            # rotation augmentation
            # do not support proposals currently
            # img_show = img.copy()
            # mmcv.imshow(img_show, win_name='original')
            for angle in [90, 180, 270]:

                for scale in cfg.data.test.img_scale:
                    _img, _img_meta,  = prepare_rotation_single(
                        img, scale, False, angle)
                    imgs.append(_img)
                    # img_metas.append(DC(_img_meta, cpu_only=True))
                    img_metas.append(_img_meta)
                    # proposals.append(_proposal)
                    # if self.flip_ratio > 0:
                    #     _img, _img_meta = prepare_rotation_single(
                    #         img, scale, True, proposal, angle)
                    #     imgs.append(_img)
                    #     img_metas.append(DC(_img_meta, cpu_only=True))
                    # # # # TODO: rm if after debug
                    # if angle == 180:
                    #     img_show = _img.cpu().numpy().copy()
                    #     mmcv.imshow(img_show, win_name=str(angle))
                    # import pdb;pdb.set_trace()

        data = dict(img=imgs, img_meta=img_metas)
        # if self.proposals is not None:
        #     data['proposals'] = proposals
        return data