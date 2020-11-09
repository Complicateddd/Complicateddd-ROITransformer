import argparse
import os.path as osp
import shutil
import tempfile
from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections,inference_detector_2
import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
import time
import numpy as np
import DOTA_devkit.polyiou as polyiou
import math
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

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def single_gpu_test(model, data_loader, show=False, log_dir=None):
    model.eval()
    results = []

    class_names=('1', '2', '3', '4', '5')
    threshold=0.1
    dataset = data_loader.dataset
    if log_dir != None:
        filename = 'inference{}.log'.format(get_time_str())
        log_file = osp.join(log_dir, filename)
        f = open(log_file, 'w')
        prog_bar = mmcv.ProgressBar(len(dataset), file=f)
    else:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        print(data['img_meta'][0].data[0][0]['filename'])
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        
        for j, name_cls in enumerate(class_names):
            dets = result[j]
            for det in dets:
                bbox = det[:8]
                score = round(det[-1],2)
                if score < threshold:
                    continue
                bbox = list(map(int, bbox))
                print(bbox)

        results.append(result)
        # print(result)
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--log_dir', help='log the inference speed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    import os
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

# if not distributed:
    model = MMDataParallel(model, device_ids=[0])
    # outputs = single_gpu_test(model, data_loader, args.show, args.log_dir)
    # else:
    #     model = MMDistributedDataParallel(model.cuda())
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir)


    model.eval()
    results = []

    result_file=open("/media/ubuntu/data/huojianjun/AerialDetection/tools/科目四_莘莘学子.txt",'w')

    class_names=('1', '2', '3', '4', '5')
    threshold=0.1
    dataset = data_loader.dataset
    if args.log_dir != None:
        filename = 'inference{}.log'.format(get_time_str())
        log_file = osp.join(log_dir, filename)
        f = open(log_file, 'w')
        prog_bar = mmcv.ProgressBar(len(dataset), file=f)
    else:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        name=data['img_meta'][0].data[0][0]['filename']

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # for i in range(5):
        #     keep = py_cpu_nms_poly_fast_np(result[i], 0.1)
        #     result[i] = result[i][keep]

        srcpath=os.path.join("/media/ubuntu/data/huojianjun/科目四初赛第一阶段/test2/images",name)

        img = draw_poly_detections(srcpath, result, class_names, scale=1, threshold=0.1)
        
        dstpath=os.path.join("/media/ubuntu/data/huojianjun/AerialDetection/demo_test",name)

        cv2.imwrite(dstpath, img)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    # return results








    # rank, _ = get_dist_info()
    # if args.out and rank == 0:
    #     print('\nwriting results to {}'.format(args.out))
    #     mmcv.dump(outputs, args.out)
    #     eval_types = args.eval
    #     if eval_types:
    #         print('Starting evaluate {}'.format(' and '.join(eval_types)))
    #         if eval_types == ['proposal_fast']:
    #             result_file = args.out
    #             coco_eval(result_file, eval_types, dataset.coco)
    #         else:
    #             if not isinstance(outputs[0], dict):
    #                 result_file = args.out + '.json'
    #                 results2json(dataset, outputs, result_file)
    #                 coco_eval(result_file, eval_types, dataset.coco)
    #             else:
    #                 for name in outputs[0]:
    #                     print('\nEvaluating {}'.format(name))
    #                     outputs_ = [out[name] for out in outputs]
    #                     result_file = args.out + '.{}.json'.format(name)
    #                     results2json(dataset, outputs_, result_file)
    #                     coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()
