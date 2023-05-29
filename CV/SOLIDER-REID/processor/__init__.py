import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm
from utils.meter import AverageMeter
from utils.metrics import Postprocessor, R1_mAP_eval


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    best_map = 0
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, pid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                # the model outputs 3 things
                # cls_score, global_feat, featmaps
                # featmaps is unused
                score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _ = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            else:
                model.eval()
                for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        # cam_label and view_label are not even used as parameters
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, pid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        if epoch % checkpoint_period == 0:
            best_map = mAP
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}_map{}_acc{}.pth'.format(epoch, mAP, acc_meter.avg)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}_map{}_acc{}.pth'.format(epoch, mAP, acc_meter.avg)))
        torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 test_loader,
                 num_query,
                 threshold,
                 output_dist_mat=False):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    postprocessor = Postprocessor(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for img, imgpath in tqdm(test_loader):
        # img shape: (num_query + num_gallery, 3, 224, 224) -> (num_query + num_gallery, channel, width, height)
        # num_query is always 1 because there is only 1 suspect for each test set image.
        # num_gallery is the number of cropped bboxes, which can be anywhere from 0 to 4.
        with torch.no_grad():
            img = img.to(device)
            feat, _ = model(img)
            postprocessor.update(feat)

    dist_mat = postprocessor.compute()
    if output_dist_mat:
        return list(dist_mat[0])

    # perform thresholding to determine which gallery image, if any, are matches with the query
    dist_mat = (dist_mat < threshold).astype(int)  # boolean array
    results = []
    for i, test_set_bbox_path in enumerate(imgpath[num_query:]):
        results.append((test_set_bbox_path, dist_mat[0][i]))
    return results


def do_batch_inference(cfg,
                       model,
                       test_loader,
                       num_query,
                       threshold,
                       output_dist_mat=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("transreid.test")
    logger.info(f"Enter {cfg.EXECUTION_MODE}")
    postprocessor = Postprocessor(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    imgpath_list = []
    camid_list = []
    for img, imgpath, camid in tqdm(test_loader):
        # img shape: (num_query + num_gallery, 3, 224, 224) -> (num_query + num_gallery, channel, width, height)
        # num_query total no of suspects, 1600
        # num_gallery is the number of cropped bboxes, for qualifiers is about 3.5k.
        # len(camid_list) and len(imgpath_list) would be num_query + num_gallery which is 1600 + ~3.5k = ~5.1k.
        with torch.no_grad():
            img = img.to(device)  # img is a tensor containing both the query and gallery images
            feat, _ = model(img)
            postprocessor.update(feat)
            imgpath_list.extend(imgpath)
            camid_list.extend(camid)
    camid_list_np_excl_query = np.array(camid_list[num_query:])  # for later vectorized np.where instead of linear search on list
    dist_mat = postprocessor.compute()  # (1600, ~3.5k)

    results = []
    if output_dist_mat:
        relevant_distances = []  # used for plotting
    else:  # perform thresholding to determine which gallery image, if any, are matches with the query
        dist_mat_bool = (dist_mat < threshold).astype(int)  # boolean array
    prev_camid = None
    same_camid_counter = 0
    for camid, test_set_bbox_path in zip(camid_list[num_query:], imgpath_list[num_query:]):  # skip the first len(query) items as they are suspect images
        if prev_camid != camid:
            same_camid_counter = 0
            prev_camid = camid
        idx = np.where(camid_list_np_excl_query == camid)[0]  # index of the gallery images in the dist_mat that correspond to a particular cam_id
        # dist_mat axis 0 is query (camid) and axis 1 is gallery. Each camid can have multiple gallery images. The same camid counter will reset to 0 when moving onto the next camid. idx is a list of indexes of the same camid, so using the counter to access the correct index in the list.
        if output_dist_mat:
            relevant_distances.append(dist_mat[camid][idx[same_camid_counter]])
        else:
            results.append((test_set_bbox_path, dist_mat_bool[camid][idx[same_camid_counter]]))
        same_camid_counter += 1
    if output_dist_mat:
        return relevant_distances
    return results


def get_distance_distributions(cfg,
                               model,
                               val_loader,
                               num_query):
    postprocessor = Postprocessor(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    device = 'cuda'
    model.to(device)
    model.eval()

    pids_running_list = []
    camids_running_list = []

    print('Inferring the validation set...')
    for imgs, pids, camids, camids_batch, viewids, img_paths in tqdm(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            feats, _ = model(imgs)
            postprocessor.update(feats)
            pids_running_list.extend(pids)
            camids_running_list.extend(camids)

    dist_mat = postprocessor.compute()

    q_pids, q_camids = np.asarray(pids_running_list[:num_query]), np.asarray(camids_running_list[:num_query])
    g_pids, g_camids = np.asarray(pids_running_list[num_query:]), np.asarray(camids_running_list[num_query:])

    # in dist_mat, query images are the rows, while gallery images are the columns
    # however in this case the query and gallery set are identical

    inter_class_distances = []
    intra_class_distances = []
    # iterate only through the lower triangle of the matrix
    for r in range(dist_mat.shape[0]):
        for c in range(r):
            if q_pids[r] == g_pids[c]:
                # means the query and gallery image are the same identity
                if q_camids[r] == g_camids[c]:
                    # means the query and gallery image are the same identity and camera
                    # this is a perfect match and we should ignore
                    continue
                else:
                    # means the query and gallery image are the same identity but different cameras
                    # this is an intra-class distance
                    intra_class_distances.append(dist_mat[r][c])
            else:
                # means the query and gallery image are different identities
                # this is an inter-class distance
                inter_class_distances.append(dist_mat[r][c])

    return inter_class_distances, intra_class_distances
