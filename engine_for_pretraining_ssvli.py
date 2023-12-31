import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import wandb
import matplotlib.pyplot as plt
import os
import textwrap
import cv2
import numpy as np
from loss_ssvli import SSVLI_Loss, SSVLI_SigLipLoss, Feature_Reconstruction_Loss


def train_one_epoch_ssvli(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, loss_weight= None, lambda_1=0, lambda_2=0, lambda_3=1, ssvli_iter=10,
                    accum_freq=1, teacher_model=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss(reduction='none')
    loss_func_ssvli = SSVLI_Loss(stu_tau=0.1,tea_tau=0.04,loss_weight=0.5, local_loss=False)
    loss_func_feature_reconstruction = Feature_Reconstruction_Loss()
    # loss_func_ssvli = SSVLI_SigLipLoss()

    if accum_freq == 1:
        for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            videos, video_texts, motion_patch_yabs, bboxs, bool_masked_pos, target = batch
            # ###
            # remvoe duplicates based on target when we use CLIP
            _, unique_idx = np.unique(target.numpy(), return_index=True)
            videos = videos[unique_idx]
            video_texts = video_texts[unique_idx]
            motion_patch_yabs = motion_patch_yabs[unique_idx]
            bboxs = bboxs[unique_idx]
            bool_masked_pos = bool_masked_pos[unique_idx]
            target = target[unique_idx]
            ###
            videos = videos.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

            #create mask on video based on the bbox
            video_masks= torch.zeros_like(videos)
            for v in range(videos.shape[0]):
                video_bbox_region = bboxs[v]
                for frame_index in range(videos.shape[1]):
                    video_masks[v][:,frame_index, int(video_bbox_region[frame_index,1]):int(video_bbox_region[frame_index,3]), int(video_bbox_region[frame_index,0]):int(video_bbox_region[frame_index,2])] = 1 # y , x
            
            mask_for_input = video_masks.clone()  

                    

            with torch.no_grad():
                # calculate the predict label
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]



                unnorm_videos = videos * std + mean  # in [0, 1]

                if normlize_target:
                    videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                    videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                        ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    ###########
                    tube_mean = videos_squeeze.mean(dim=-2, keepdim=True)
                    tube_std = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                    ###########
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                    # update video masks 
                    video_masks = rearrange(video_masks, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                    video_masks = rearrange(video_masks, 'b n p c -> b n (p c)')
                else:
                    videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                    # update video masks 
                    video_masks = rearrange(video_masks, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

                B, _, C = videos_patch.shape
                labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
                # create label mask for applying bbox
                mask_labels = video_masks[bool_masked_pos].reshape(B, -1, C)

                
                ### mask the input video (put 0 for the pixels outside of the BB)
                # videos = videos * mask_for_input

                # # find zero elements   in labels
                # labels_mask_loc = torch.where(mask_labels==0)
                # labels_mask = torch.ones_like(labels)
                # labels_mask[labels_mask_loc[0], labels_mask_loc[1], labels_mask_loc[2]] = loss_weight
            
        
            with torch.cuda.amp.autocast():
                # videos = torch.zeros_like(videos)
                # videos[:, :, 2:4, 0*16:1*16, 0*16:1*16] = 1
                # model forward
                outputs, embedded_patches, mapped_embedded_patches, pred_features, _, mapped_masked_pred_features, logit_scale = model(videos, bool_masked_pos)
                # teacher model forward
                with torch.no_grad():
                    _, _, _, _, mapped_masked_embedded_patches, _, _ = teacher_model(videos, bool_masked_pos)
                    mapped_masked_embedded_patches = F.layer_norm(mapped_masked_embedded_patches, (mapped_masked_embedded_patches.size(-1),))  # normalize over feature-dim

                loss = lambda_1 * loss_func(input=outputs, target=labels) 
                ssvli_loss = 0
                # repeat the motion_patch_yabs for 8 times
                motion_patch_yabs = motion_patch_yabs.repeat(1,8)
                ssvli_acc_list = []
                for i in range (0, ssvli_iter):
                    # find one element indexes in motion_patch_yabs
                    random_index = []
                    vid_embed = []
                    x, y = torch.where(motion_patch_yabs==1)
                    for j in range(B):
                        x_loc = torch.where(x==j)[0].numpy()
                        # shuffle list x_loc
                        np.random.shuffle(x_loc)
                        # randomly select one element from the list
                        if len(x_loc) > 0:
                            random_index.append([j, y[x_loc[0]]])
                        else:
                            random_index.append([j, np.random.randint(0, 1536)])

                    random_index_patch = torch.tensor(random_index)

                    # get the random index for each video
                    video_embed = mapped_embedded_patches[random_index_patch[:,0], random_index_patch[:,1], :] # for patch-wise
                    # video_embed = embedded_patches.mean(dim=1) # for average patch

                    ############################ inside bbox average
                    # # randomly select one element from the list
                    #     if len(x_loc) > 0:
                    #         vid_embed.append(mapped_embedded_patch[j, y[x_loc], :].mean(dim=0))
                    #     else:
                    #         vid_embed.append(mapped_embedded_patch[j, np.array(list(range(1536))), :].mean(dim=0))

                    # # get the random index for each video
                    # video_embed = torch.stack(vid_embed, dim=0)
                    ############################


                    ssvli_input_dict = {'video_embed': video_embed, 'text_embed': video_texts.squeeze(1).to(device), 
                                        'motion_patch_yabs': motion_patch_yabs, 'logit_scale': logit_scale,
                                        'bbox': bboxs, 'bool_masked_pos': bool_masked_pos}
                    ssvli = loss_func_ssvli(ssvli_input_dict)
                    ## if the loss is nan, then skip this iteration
                    # if math.isnan(ssvli['loss']):
                    #     continue
                    ssvli_loss = ssvli_loss + lambda_2 * ssvli['loss']
                    ssvli_acc_list.append(ssvli['clip_patch_wise_acc'])

                FR_loss = lambda_3 * loss_func_feature_reconstruction(mapped_masked_embedded_patches, mapped_masked_pred_features)['loss']

                ssvli_loss = ssvli_loss / ssvli_iter
                ssvli_patch_wise_acc = sum(ssvli_acc_list) / len(ssvli_acc_list)


                # apply label mask to loss and average
                # loss = loss * labels_mask
                mse_loss = loss.mean()
                loss = mse_loss +  ssvli_loss/B + FR_loss/B
            

            loss_value_total = loss.item()
            loss_value_MSE = mse_loss.item()
            loss_value_SSVLI = (ssvli_loss/B).item()
            loss_value_FR = (FR_loss/B).item()


            if not math.isfinite(loss_value_total):
                print("Loss is {}, stopping training".format(loss_value_total))
                sys.exit(1)

            optimizer.zero_grad()
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            metric_logger.update(loss_total=loss_value_total)
            metric_logger.update(loss_MSE=loss_value_MSE)
            metric_logger.update(loss_SSVLI=loss_value_SSVLI)
            metric_logger.update(loss_FR=loss_value_FR)
            metric_logger.update(patch_wise_acc=ssvli_patch_wise_acc)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            # # log to weights & biases
            # wandb_dict = {}
            # for key, value in metric_logger.meters.items():
            #     wandb_dict["train_iter_"+key] = value.global_avg
            # wandb.log(wandb_dict, step=it)

            if log_writer is not None:
                log_writer.update(loss_total=loss_value_total, head="loss_total")
                log_writer.update(loss_MSE=loss_value_MSE, head="loss_MSE")
                log_writer.update(loss_SSVLI=loss_value_SSVLI, head="loss_SSVLI")
                log_writer.update(loss_FR=loss_value_FR, head="loss_FR")
                log_writer.update(patch_wise_acc=ssvli_patch_wise_acc, head='patch_wise_acc')
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step)
    else:
        # accumulate gradients for multiple steps #### dataloader 5000 batch size 10 accum_freq 50 => num_batches_per_epoch = 10
        # num_batches_per_epoch = data_loader.num_batches // accum_freq
        accum_videos, accum_texts, accum_labels, accum_motion_patch_yabs, accum_features = [], [], [], [], {}
        accum_random_index_patch = []
        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            i_accum = i // accum_freq
            # step = num_batches_per_epoch * epoch + i_accum

            # assign learning rate & weight decay for each step
            it = start_steps + i_accum  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for _, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            videos, video_texts, motion_patch_yabs, bboxs, bool_masked_pos, target = batch
            videos = videos.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

            #create mask on video based on the bbox
            video_masks= torch.zeros_like(videos)
            for v in range(videos.shape[0]):
                video_bbox_region = bboxs[v]
                for frame_index in range(videos.shape[1]):
                    video_masks[v][:,frame_index, int(video_bbox_region[frame_index,1]):int(video_bbox_region[frame_index,3]), int(video_bbox_region[frame_index,0]):int(video_bbox_region[frame_index,2])] = 1 # y , x            

            with torch.no_grad():
                # calculate the predict label
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

                unnorm_videos = videos * std + mean  # in [0, 1]

                if normlize_target:
                    videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                    videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                        ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    ###########
                    tube_mean = videos_squeeze.mean(dim=-2, keepdim=True)
                    tube_std = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                    ###########
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                    # update video masks 
                    video_masks = rearrange(video_masks, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                    video_masks = rearrange(video_masks, 'b n p c -> b n (p c)')
                else:
                    videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                    # update video masks 
                    video_masks = rearrange(video_masks, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

                B, _, C = videos_patch.shape
                labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
                # create label mask for applying bbox
                mask_labels = video_masks[bool_masked_pos].reshape(B, -1, C)

                
                ### mask the input video (put 0 for the pixels outside of the BB)
                # videos = videos * mask_for_input

                # # find zero elements   in labels
                # labels_mask_loc = torch.where(mask_labels==0)
                # labels_mask = torch.ones_like(labels)
                # labels_mask[labels_mask_loc[0], labels_mask_loc[1], labels_mask_loc[2]] = loss_weight
            
        

            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs, embedded_patches = model(videos, bool_masked_pos)
                    
                    # store the data
                    accum_videos.append(videos)
                    accum_texts.append(video_texts.squeeze(1).to(device))
                    accum_labels.append(labels)
                    accum_motion_patch_yabs.append(motion_patch_yabs)
                    ##
                    motion_patch_yab = motion_patch_yabs.repeat(1,8)
                    random_index_patch_list = []
                    for m in range (0, ssvli_iter):
                        # find one element indexes in motion_patch_yabs
                        random_index = []
                        x, y = torch.where(motion_patch_yab==1)
                        for j in range(B):
                            x_loc = torch.where(x==j)[0].numpy()
                            # shuffle list x_loc
                            np.random.shuffle(x_loc)
                            # randomly select one element from the list
                            if len(x_loc) > 0:
                                random_index.append([j, y[x_loc[0]]])
                            else:
                                random_index.append([j, np.random.randint(0, 1536)])

                        random_index_patch_list.append(torch.tensor(random_index))
                    ##
                    accum_random_index_patch.append(random_index_patch_list)

                    # accumulate the features
                    model_output = {
                        "video_embed": [embedded_patches[random_index_patch_list[x][:,0], random_index_patch_list[x][:,1], :].detach().clone() for x in range(ssvli_iter)],
                        "text_embed": [video_texts.squeeze(1).to(device).detach().clone() for x in range(ssvli_iter)],
                    }
                    for k, v in model_output.items():
                        if k not in accum_features:
                            accum_features[k] = []
                        accum_features[k].append(v)

            if ((i + 1) % accum_freq) > 0:
                continue

            # free cuda memory
            torch.cuda.empty_cache()

            # Now, compute the loss and backprop.
            optimizer.zero_grad()
            losses = []
            for j in range(accum_freq):
                images = accum_videos[j]
                texts = accum_texts[j]
                with torch.cuda.amp.autocast():
                    outputs, embedded_patches = model(videos, bool_masked_pos)

                    # compute MAE loss
                    loss = lambda_1 * loss_func(input=outputs, target=accum_labels[j])

                    # repeat the motion_patch_yabs for ssvli_iter times
                    ssvli_loss = 0
                    ssvli_acc_list = []
                    for n in range (0, ssvli_iter):
                        # video_embed = embedded_patches[accum_random_index_patch[j][n][:,0], accum_random_index_patch[j][n][:,1], :]
                        video_embed = embedded_patches.mean(dim=1)
                        model_output = {
                            "video_embed": video_embed,
                            "text_embed": accum_texts[j],
                        }

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(
                                [x[n] for x in accumulated[:j]] + 
                                [model_output[key]] + 
                                [x[n] for x in accumulated[j + 1:]]
                                )
                        l = loss_func_ssvli(inputs)
                        ssvli_loss = ssvli_loss + lambda_2 * l['loss']
                        ssvli_acc_list.append(l['clip_patch_wise_acc'])
                        del inputs
                    
                    ssvli_loss = ssvli_loss / ssvli_iter
                    ssvli_patch_wise_acc = sum(ssvli_acc_list) / len(ssvli_acc_list)

                    loss = loss.mean() + ssvli_loss#/B
            

                loss_value = loss.item()

                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=False)


            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            grad_norm = loss_scaler.update(optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(),)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(patch_wise_acc=ssvli_patch_wise_acc)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            # # log to weights & biases
            # wandb_dict = {}
            # for key, value in metric_logger.meters.items():
            #     wandb_dict["train_iter_"+key] = value.global_avg
            # wandb.log(wandb_dict, step=it)

            if log_writer is not None:
                log_writer.update(loss_total=loss_value_total, head="loss_total")
                log_writer.update(loss_MSE=loss_value_MSE, head="loss_MSE")
                log_writer.update(loss_SSVLI=loss_value_SSVLI, head="loss_SSVLI")
                log_writer.update(loss_FR=loss_value_FR, head="loss_FR")
                log_writer.update(patch_wise_acc=ssvli_patch_wise_acc, head='patch_wise_acc')
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step)

            # reset the accum
            accum_videos, accum_texts, accum_labels, accum_motion_patch_yabs, accum_features = [], [], [], [], {}
            accum_random_index_patch = []

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_predictive(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, accum_freq=1, teacher_model=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, video_texts, motion_patch_yabs, bboxs, bool_masked_pos, target = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_masked_pos[:,0:196*2] = False #0:196*2 means the first 4 frames from 16 frames are unmasked
 
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                ###########
                tube_mean = videos_squeeze.mean(dim=-2, keepdim=True)
                tube_std = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                ###########
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs, embedded_patches, mapped_embedded_patches, pred_features, _, mapped_masked_pred_features, logit_scale = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

