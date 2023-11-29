import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from loss_ssvli_utils import (get_rank,
                                get_world_size, 
                                all_gather_batch,
                                neighbour_exchange_bidir_with_grad,
                                neighbour_exchange_with_grad)

from utils import AllReduce


class SSVLI_Loss(nn.Module):
    def __init__(
        self,
        stu_tau=0.1,
        tea_tau=0.04,
        loss_weight=0.5,
        local_loss=True,
        ):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.world_size = get_world_size()
        self.local_loss = local_loss
        
        self.stu_tau = stu_tau
        self.tea_tau = tea_tau
        self.loss_weight = loss_weight

    def forward(self, outputs):
        video_embed = outputs['video_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = video_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            if self.local_loss:
                self.labels = torch.arange(
                    local_batch_size, device=video_embed.device
                )
            else:
                self.labels = torch.arange(
                    self.world_size*local_batch_size, device=video_embed.device
                )
            self.last_local_batch_size = local_batch_size

        # normalized features
        video_embed = F.normalize(video_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        video_embed_all, text_embed_all = \
            all_gather_batch([video_embed, text_embed])

        # cosine similarity as logits
        if self.local_loss:
            logits_per_video = logit_scale * video_embed @ text_embed_all.t()
            logits_per_text = logit_scale * text_embed @ video_embed_all.t()
        else:
            logits_per_video = logit_scale * video_embed_all @ text_embed_all.t()
            logits_per_text = logits_per_video.t()

        # compute loss
        clip_loss_patch_wise = (F.cross_entropy(logits_per_video, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_video, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size
   
        # loss = AllReduce.apply(clip_loss_patch_wise)
        loss = clip_loss_patch_wise

        return {'loss': loss, 'clip_patch_wise_acc': acc}
    
class SSVLI_SigLipLoss(nn.Module):
    def __init__(
        self,
        cache_labels=False,
        bidir=True,
        use_horovod=False,
        ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = get_rank()
        self.world_size = get_world_size()
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = None
        self.last_local_batch_size = None

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss
       
    def forward(self, outputs, output_dict=False):
        video_embed = outputs['video_embed']
        text_embed = outputs['text_embed']
        logit_scale = self.logit_scale.exp()
        local_batch_size = video_embed.size(0)

        # normalized features
        video_embed = F.normalize(video_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        
        loss = self._loss(video_embed, text_embed, logit_scale,)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_embed
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            video_embed,
                            f,
                            logit_scale,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        video_embed,
                        text_features_recv,
                        logit_scale,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_embed
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        video_embed,
                        text_features_from_left,
                        logit_scale,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        # compute accuracy
        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=video_embed.device
            )
            self.last_local_batch_size = local_batch_size

        with torch.no_grad():
            pred = torch.argmax(self.get_logits(video_embed, text_embed, logit_scale,), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_patch_wise_acc': acc}
    



class Feature_Reconstruction_Loss(nn.Module):
    def __init__(
        self
        ):
        super().__init__()
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.KLDivLoss()
        # self.loss = nn.MSELoss()

    def forward(self, input, target):
        # input = input.softmax(dim=-1)
        # target = target.softmax(dim=-1)
        # input = F.log_softmax(input, dim=-1)
        # target = F.softmax(target, dim=-1)
        # loss = self.loss(input, target)

        loss = F.smooth_l1_loss(input, target)
        loss = AllReduce.apply(loss)

        return {'loss': loss}