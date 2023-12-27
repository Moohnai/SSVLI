from turtle import pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transforms import *
import torchvision.transforms.functional as F
import numpy as np
import os
import cv2
import decord
import orjson
from PIL import Image
from masking_generator import TubeMaskingGenerator, TubeMaskingGenerator_BB
import math
import torchvision.transforms._transforms_video as transforms_video
import pandas as pd
from decord import VideoReader, cpu
import csv
import torch.nn as nn
import os.path as osp

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()

def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader

def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0'

    vr = get_video_reader(
        osp.join(root, '{}'.format(vid)),
        num_threads=threads,
        fast_rrc=fast_rrc, rrc_params=rrc_params,
        fast_rcc=fast_rcc, rcc_params=rcc_params,
    )
    end_second = min(end_second, len(vr) / fps)

    # calculate frame_ids
    frame_offset = int(np.round(second * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

    # load frames
    assert max(frame_ids) < len(vr)
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
    except decord.DECORDError as error:
        print(error)
        frames = vr.get_batch([0] * len(frame_ids)).asnumpy()

    return torch.from_numpy(frames.astype(np.float32)), frame_ids

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)

class DataAugmentationForVideoMAE_BB(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop_BB_no_global_union(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            # ToTorchFormatTensor(div=False),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, process_bbx = self.transform(images)
        return process_data, process_bbx, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class VideoMAE_ssvli(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 patch_size=16,
                 patch_yab_strategy = 'fully_included' # 'fully_included' or 'partially_included'
                 ):

        super(VideoMAE_ssvli, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.patch_size = patch_size
        self.patch_yab_strategy = patch_yab_strategy
        # self.video_text = torch.load('/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_action_text.pt')
        # self.video_text = torch.load("/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_video_captions.pt")
        # self.video_text = torch.load("/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_image_captions.pt")
        self.video_text = torch.load("/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_mixed_captions.pt")
        self.video_text = [x.float().to('cpu') for x in self.video_text]
        Total_video_BB_no_global_union={}
        print("Loading bbox json file...")
        with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/EPIC_100_BB_smooth_train.json', "r", encoding="utf-8") as f:
            Total_video_BB = orjson.loads(f.read())
        self.bb_data = Total_video_BB

        # SSV2
        # with open('/home/mona/VideoMAE/SSV2_BB/bounding_box_smthsmth_scaled.json', "r", encoding="utf-8") as f:
        #     Total_video_BB = orjson.loads(f.read())
        # self.bb_data = Total_video_BB


        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

    # ###fuction for visualizing image with bounding box
    def visual_union_bbx_Epic_Kitchens (self, images, union_bbx, video_name=""):
        """
        images (torch.Tensor or np.array): list of images in torch or numpy type.
        union_bbx (List[List]): list of list having union bounding box of each frame in [x1, y1, x2, y2]
        format of json file for bbx: a dictionary with key as video name and value as a list of bbx for each frame. 
        each element of this list is a dictionary with key as 'labels' and value as a list that include a dictionary with two key 
        'box2d' that its value is a dictionary with key as 'x1', 'x2', 'y1', 'y2' 
        and another key is 'gt_annotation' that its value here is 'union'

        for instance: Total_video_BB ['video_0'][0]['labels'][0]['box2d']>>>>>>>>>{'x1': 0, 'y1': 0, 'x2': 570, 'y2': 320}
        """
        if isinstance(images, torch.Tensor):
            images = images.view((16, 3) + images.size()[-2:])
        color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255,255)]
        if not os.path.exists('VideoMAE/Epic_Kitchens_BB/data/visual_bbx'):
            os.makedirs('VideoMAE/Epic_Kitchens_BB/data/visual_bbx')
        for i, (img, bbx) in enumerate(zip(images, union_bbx)):
            if isinstance(img, torch.Tensor):
                frame = img.numpy().astype(np.uint8).transpose(1, 2, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif isinstance(img, Image.Image):
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            bb = bbx.copy().reshape(-1)
            (x1,y1,x2,y2) = (bb[0], bb[1], bb[2], bb[3])
        
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[0], 4)
            # for c, b in enumerate(bbx):
            #     cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color_list[c], 4)

            cv2.imwrite(f'VideoMAE/Epic_Kitchens_BB/data/visual_bbx/{video_name}_{i}.png', frame)   
       



    # def visual_bbx (self, images, bboxes):
    #     """
    #     images (torch.Tensor or np.array): list of images in torch or numpy type.
    #     bboxes (List[List]): list of list having bounding boxes in [x1, y1, x2, y2]
    #     """
    #     if isinstance(images, torch.Tensor):
    #         images = images.view((16, 3) + images.size()[-2:])
    #     color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255,255)]
    #     if not os.path.exists('VideoMAE/SSV2_BB/data/visual_bbx'):
    #         os.makedirs('VideoMAE/SSV2_BB/data/visual_bbx')
    #     for i, (img, bbx) in enumerate(zip(images, bboxes)):
    #         if isinstance(img, Image.Image):
    #             frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #             # (x1,y1,x2,y2) = (bbx[0], bbx[1], bbx[2], bbx[3])
    #         elif isinstance(img, torch.Tensor):
    #             frame = img.numpy().astype(np.uint8).transpose(1, 2, 0)
    #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             # if len(bbx) != 0:
    #             #     (x1,y1,x2,y2) = (bbx[0][0], bbx[0][1], bbx[0][2], bbx[0][3])
    #         if len(bbx) != 0:
    #             # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[0], 4)
    #             ##
    #             for c, b in enumerate(bbx):
    #                 cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color_list[c], 4)
    #             ##
    #         cv2.imwrite(f'VideoMAE/SSV2_BB/data/visual_bbx/{i}.png', frame)


    def __getitem__(self, index):

        directory, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images ,frame_id_list = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
        bboxs = []
        bboxs_labels = []
        # union_frames_bbox = []
        frames_bbox = []
        ### for EPIC-KITCHENS
        for idx, c in enumerate(frame_id_list):
            union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
            frames_bbox.append(union_frame_bboxs)

        # for idx, c in enumerate(frame_id_list):
        #     bboxs.append([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]) # x1, y1, x2, y2
        #     bboxs_labels.append([x['gt_annotation'] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']])
        #     union_frame_bboxs = np.array(bboxs[-1])
        #     if len(union_frame_bboxs) == 0:
        #         union_frame_bboxs = np.array([[0, 0, 1, 1]])
        #     union_frame_bboxs = np.array([np.min(union_frame_bboxs[:, 0]), np.min(union_frame_bboxs[:, 1]), np.max(union_frame_bboxs[:, 2]), np.max(union_frame_bboxs[:, 3])])
        #     # if all the bboxs are zero or have equal values in x and y, then we use the whole image as the bbox
        #     if union_frame_bboxs[0] == union_frame_bboxs[2] or union_frame_bboxs[1] == union_frame_bboxs[3]:
        #         union_frame_bboxs = np.array([0, 0, 1, 1])
        #     frames_bbox.append(union_frame_bboxs)

        frames_bbox = np.array(frames_bbox)  # x1, y1, x2, y2

        # create a union bbox of all the frames
        union_bbx = np.array([np.min(frames_bbox[:, 0]), np.min(frames_bbox[:, 1]), np.max(frames_bbox[:, 2]), np.max(frames_bbox[:, 3])])
        frames_bbox = [union_bbx]*len(frames_bbox)


        # self.visual_union_bbx_Epic_Kitchens(images, frames_bbox)

        try:
            process_data, process_bbx, mask = self.transform((images, frames_bbox)) # T*C,H,W
        except Exception as e:
            print(f"error in video {video_name}")
            raise e

        ####moved it to GroupMultiScaleCrop_BB_no_global_union
        # process_bbx_filtered= []
        # for bbx in process_bbx:
        #     if len(bbx)>0:
        #         process_bbx_filtered.append(np.array([bbx[0][0], bbx[0][1], bbx[0][2], bbx[0][3]]))
        #     elif len(bbx)<=0:
        #         process_bbx_filtered.append(np.array([0, 0, 1, 1]))
        # process_bbx = np.array(process_bbx_filtered)


        # # if the object bbox is removed in the process of transform
        # if len(process_bbx) == 0:
        #     bbox = np.array([0, 0, process_data.size()[-2], process_data.size()[-1]])
      
        # else:
        #     # bbox = np.array([np.min(process_bbx[:, 0]), np.min(process_bbx[:, 1]), np.max(process_bbx[:, 2]), np.max(process_bbx[:, 3])]) # x1, y1, xk, yk
        #     bbox = process_bbx
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        video_text = self.video_text[index]
        # # cast video_text into float 32 bit
        # video_text = video_text.float().to(process_data.device)
        
        #create a matrix with the size of the image and fill it with 1 in the bbox area
        motion_patch_yab_size = [ process_data.size()[-2]//self.patch_size[0], process_data.size()[-1]//self.patch_size[1]]
        motion_patch_yab = torch.zeros(motion_patch_yab_size[-2], motion_patch_yab_size[-1])
        union_bbx = process_bbx[0]
        if self.patch_yab_strategy == 'partially_included':
            x_start = math.ceil(union_bbx[0]/self.patch_size[0])
            x_end = math.floor(union_bbx[2]/self.patch_size[0])
            y_start = math.ceil(union_bbx[1]/self.patch_size[1])
            y_end = math.floor(union_bbx[3]/self.patch_size[1])
            motion_patch_yab[x_start:x_end-1, y_start:y_end-1] = 1
            
        if self.patch_yab_strategy == 'fully_included':
            x_start = math.floor(union_bbx[0]/self.patch_size[0])
            x_end = math.ceil(union_bbx[2]/self.patch_size[0])
            y_start = math.floor(union_bbx[1]/self.patch_size[1])
            y_end = math.ceil(union_bbx[3]/self.patch_size[1])
            motion_patch_yab[x_start:x_end-1, y_start:y_end-1] = 1


        return (process_data, video_text, motion_patch_yab.transpose(1, 0).flatten(), torch.LongTensor(process_bbx), mask, target)
 
    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[-1].split('\n')[0])
                # target = int(line_info[1])
                # if target > 9:
                #     continue
                item = (clip_path, target)
                clips.append(item)

    
        a=[x[1] for x in clips]
        a_unique = list(set(a))
        a_unique.sort()
        counter = {x:0 for x in a_unique}
        for x in a:
            counter[x] += 1
        # sort the dictionary based on the values
        counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
        #save in a text file
        with open('/home/mona/test_whole.txt', 'w') as f:
            for key, value in counter.items():
                f.write('%s:%s\n' % (key, value))

        # classes that have more than 90 and less than 110 videos

        b_unique = [x for x in a_unique if counter[x] > 90 and counter[x] < 110] #3510data_35classes
        clips_middle = [x for x in clips if x[1] in b_unique]
    

        b=[x[1] for x in clips_middle]
        b_unique = list(set(b)) 
        b_unique.sort()
        counter = {x[1]:0 for x in clips_middle}
        for x in b:
            counter[x] += 1
        # sort the dictionary based on the values
        counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
        #save in a text file
        with open('/home/mona/test_middle.txt', 'w') as f:
            for key, value in counter.items():
                f.write('%s:%s\n' % (key, value))
        

        return clips_middle #only 1000 videos are considered for training from the middle classes

    def _sample_train_indices(self, num_frames):
        # fix random seed for each video
        np.random.seed(10)
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list, frame_id_list

class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, data_root, anno_path, mode, metadata=None, is_trimmed=True, args=None):
        self.dataset = dataset
        self.data_root = data_root
        self.metadata = metadata
        self.is_trimmed = is_trimmed
        self.mode = mode
        self.anno_path = anno_path

        m = "validation" if mode == "test" else mode
        self.data_root = os.path.join(self.data_root, m)
        
        self.samples = []
        with open(self.anno_path) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for i, row in enumerate(csv_reader):
                pid, vid = row[1:3]
                narration = row[8]
                start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                verb, noun = int(row[10]), int(row[12])
                vid_path = 'video_{}.mp4'.format(i)

                if mode != "train":
                    # check if (verb, noun) is in the mapping
                    if f"{verb}:{noun}" not in args.mapping_vn2act:
                        continue
                
                if len(row) == 16:
                    fps = float(row[-1])
                else:
                    vid_path = os.path.join(self.data_root, vid_path)
                    vr = VideoReader(vid_path, num_threads=1, ctx=cpu(0))
                    fps = vr.get_avg_fps()
                # start_frame = int(np.round(fps * start_timestamp))
                # end_frame = int(np.ceil(fps * end_timestamp))
                self.samples.append((os.path.join(self.data_root, vid_path), start_timestamp, end_timestamp, fps, narration, verb, noun))


        if mode == "train":
            # count the number of each class
            num_each_class = {}
            for _, _, _, _, _, verb, noun in self.samples:
                if args.mapping_vn2act[f"{verb}:{noun}"] in num_each_class:
                    num_each_class[args.mapping_vn2act[f"{verb}:{noun}"]] += 1
                else:
                    num_each_class[args.mapping_vn2act[f"{verb}:{noun}"]] = 1

            selected_classes = [x for x, v in num_each_class.items() if v > 90 and v < 110] #3510data_35classes

            # fileter the samples based on the selected classes
            self.samples = [x for x in self.samples if args.mapping_vn2act[f"{x[5]}:{x[6]}"] in selected_classes]

            # update the mapping
            vn_list = [f"{x[5]}:{x[6]}" for x in self.samples]
            args.mapping_vn2act = {x: i for i, x in enumerate(set(vn_list))}
            args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in args.mapping_vn2act.items()}
            args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in args.mapping_vn2act.items()}
            args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})

            ########################################
            mapping_class_t=[]
            #count the number of samples for each class in the selected training set
            for i in range(len(args.actions)):
                mapping_class_t.append(len([x for x in self.samples if args.mapping_vn2act[f"{x[5]}:{x[6]}"] == i]))
            # print("number of samples for each class in the selected training set: ", mapping_class_t)
            #number of samples for each class in the selected training set: 
            #[107, 98, 101, 97, 97, 97, 104,
            #  99, 94, 104, 100, 98, 104, 94,
            #  109, 92, 108, 94, 106, 104, 96,
            #  92, 106, 95, 105, 108, 95, 105,
            #  103, 95, 101, 94, 99, 100, 109]             

    #########count the number of samples for each class in the selected validation set
        if mode != "train":
            mapping_class_v=[]
            for _, _, _, _, _, verb, noun in self.samples:
                if f"{verb}:{noun}" in args.mapping_vn2act:
                    for i in range(len(args.actions)):
                        mapping_class_v.append(len([x for x in self.samples if args.mapping_vn2act[f"{x[5]}:{x[6]}"] == i]))
                    # print("number of samples for each class in the selected validation set: ", mapping_class_v)
                    # number of samples for each class in the selected validation set: 
                    #[8, 12, 7, 12, 4, 10, 13,
                    #  30, 27, 14, 18, 21, 53, 6,
                    #  21, 20, 7, 16, 20, 10, 1,
                    #  24, 31, 13, 6, 11, 4, 14,
                    #  2, 9, 30, 2, 17, 4, 9]

                    #506
        ########################################
        # add fps to the annotation file
        if len(row) == 15:
            df = pd.read_csv(self.anno_path)
            df['fps'] = [self.samples[i][3] for i in range(len(self.samples))]
            df.to_csv(self.anno_path, index=False)

    def get_raw_item(
        self, i, is_training=True, num_clips=1,
        chunk_len=300, clip_length=32, clip_stride=2,
        sparse_sample=False,
        narration_selection='random',
        threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False, rcc_params=(224,),
    ):

        vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
        end_second = end_second - start_second
        start_second = 0
        frames, frame_ids = video_loader(self.data_root, vid_path, 'MP4',
                                start_second, end_second,
                                chunk_len=chunk_len, fps=fps,
                                clip_length=clip_length,
                                threads=threads,
                                fast_rrc=fast_rrc,
                                rrc_params=rrc_params,
                                fast_rcc=fast_rcc,
                                rcc_params=rcc_params,
                                jitter=is_training)
        return frames, '{}:{}'.format(verb, noun), frame_ids

    def __len__(self):
        return len(self.samples)

class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, anno_path, metadata=None, transform=None,
        is_training=True,
        num_clips=1,
        chunk_len=-1,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True,
        mode='train',
        args=None,
        root="/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/",
        ):
        super().__init__(dataset, root, anno_path, mode, metadata, is_trimmed=is_trimmed, args=args)

        self.transform = transform
        self.is_training = True if mode == 'train' else False
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample
        self.mode = mode
        self.label_mapping = args.mapping_vn2act

        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]

        if self.mode == "train":
            base_train_transform_ls = [
                Permute([3, 0, 1, 2]),
                torchvision.transforms.RandomResizedCrop(rcc_params[0], scale=(0.5, 1.0)),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
            self.transform = torchvision.transforms.Compose(base_train_transform_ls)
        else:
            base_val_transform_ls = [
                Permute([3, 0, 1, 2]),
                torchvision.transforms.Resize(rcc_params[0]),
                torchvision.transforms.CenterCrop(rcc_params[0]),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
            self.transform = torchvision.transforms.Compose(base_val_transform_ls)

    def __getitem__(self, i):
        frames, label, _ = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                
                label = self.label_mapping[label]

        return frames, label, self.samples[i][0].split("/")[-1].split(".")[0], {}

def build_pretraining_dataset_ssvli(args, patch_size=16):
    transform = DataAugmentationForVideoMAE_BB(args)
    dataset = VideoMAE_ssvli(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        patch_size=patch_size,)
    # dataset = build_dataset(is_train=True, test_mode=False, args=args)
    print("Data Aug = %s" % str(transform))
    return dataset

def build_finetuning_dataset_ssvli(is_train, test_mode, args):
    if args.data_set == 'Epic-Kitchens':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'EPIC_100_train.csv')
            num_clips = args.num_segments
            # anno_path = args.data_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'EPIC_100_validation.csv') 
            num_clips = args.test_num_segment
            
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'EPIC_100_validation.csv')
            # anno_path = args.eval_data_path 
            num_clips = args.test_num_segment


        dataset = VideoClassyDataset(
            args.data_set, 
            anno_path=anno_path,
            num_clips=num_clips,
            clip_length=args.num_frames, 
            clip_stride=args.sampling_rate,
            threads=1,
            mode=mode,
            args=args,
            )
        
        nb_classes = args.nb_classes   
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes