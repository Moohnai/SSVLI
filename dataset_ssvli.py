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
        self.video_text = torch.load('/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_action_text.pt')
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

        # classes that have more than 30 and less than 1000 videos

        b_unique = [x for x in a_unique if counter[x] > 100 and counter[x] < 300]
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



