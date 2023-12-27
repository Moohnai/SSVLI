import pandas as pd
import csv
import os 
import json
import itertools


# root addresses

root_add_train = "/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train.csv"
root_add_val = "/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_validation.csv" 
video_mp4_root_add_train = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/train"
video_mp4_root_add_val = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/validation"




# # create dataframe
# train_df = {'path':[], 'label_name':[], 'label_num':[]}
# val_df = {'path':[], 'label_name':[], 'label_num':[]}


# # ###### crete json file for labels
# # ###verb
# # train_df = pd.read_csv(root_add_train)
# # num_uniq_verb_labels = len(train_df['verb_class'].unique())
# # verb_dict = {label:[] for label in range(num_uniq_verb_labels)}
# # for i, item in train_df.iterrows():
# #     if not any(item['verb'] in verb for verb in verb_dict[item['verb_class']]):
# #         verb_dict[item['verb_class']].append(item['verb'])
# # #find verbs with same class


# # verb_dict = dict(sorted(verb_dict.items(), key=lambda item: item[0]))
# # verb_dict = {str(k): v for k, v in verb_dict.items()}

# # root_add = "/home/mona/VideoMAE/dataset/Epic_kitchen/annotation/verb"

# # with open(os.path.join(root_add, 'labels','labels.json'), 'w') as fp:
# #     json.dump(verb_dict, fp, indent=4)


# class_list = [2,3,4,7,8,9,10,14,21,26,32,47,55,63,77]

# class_dic = {2:0, 3:1, 4:2, 7:3, 8:4, 9:5, 10:6, 14:7, 21:8, 26:9, 32:10, 47:11, 55:12, 63:13, 77:14}
# #high accuracy classes: 2,3,4,7,10

# train_label = pd.read_csv(root_add_train)
# for i, item in train_label.iterrows():
#     if item['noun_class'] not in class_list:
#         continue
#     else:
#         path = os.path.join(video_mp4_root_add_train, f"video_{i}.mp4")
#         if not os.path.exists(path):
#             continue
#         label_name = item ['noun']
#         label_num = item ['noun_class']
#         # label_name = item ['noun']
#         # label_num = item ['noun_class']
#         train_df['path'].append(path)
#         train_df['label_name'].append(label_name)
#         train_df['label_num'].append(class_dic[label_num])
        

# val_label = pd.read_csv(root_add_val)
# for i, item in val_label.iterrows():
#     if item['noun_class'] not in class_list:
#         continue
#     else:
#         path = os.path.join(video_mp4_root_add_val, f"video_{i}.mp4")
#         if not os.path.exists(path):
#             continue
#         label_name = item ['noun']
#         label_num = item ['noun_class']
#         # label_name = item ['noun']
#         # label_num = item ['noun_class']
#         val_df['path'].append(path)
#         val_df['label_name'].append(label_name)
#         val_df['label_num'].append(class_dic[label_num])


# train_df = pd.DataFrame(train_df)
# val_df = pd.DataFrame(val_df)


# # to_csv() 
# csv_annotation_root = "/home/mona/SSVLI/dataset/epic_kitchens/annotation/noun/15classes"
# if not os.path.exists(csv_annotation_root):
#     os.makedirs(csv_annotation_root)
# train_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "train.csv"), sep=' ', na_rep='', float_format=None, 
# columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
# compression='infer', quoting=None, quotechar='"', 
# chunksize=None, date_format=None, doublequote=True, escapechar=None, 
# decimal='.', errors='strict', storage_options=None)

# val_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "val.csv"), sep=' ', na_rep='', float_format=None, 
# columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
# compression='infer', quoting=None, quotechar='"', 
# chunksize=None, date_format=None, doublequote=True, escapechar=None, 
# decimal='.', errors='strict', storage_options=None)

# # number of train and val samples
# print(f"number of train samples: {len(train_df)}")
# print(f"number of val samples: {len(val_df)}")

# # number of train and val samples per class
# print(f"number of train samples per class: {train_df['label_num'].value_counts()}")
# print(f"number of val samples per class: {val_df['label_num'].value_counts()}")


# # # #######
# # number of train samples: 23751
# # number of val samples: 3681
# # number of train samples per class:
# # 2     6927
# # 3     4870
# # 4     3483
# # 8     1861
# # 7     1742
# # 9     1595
# # 10    1574
# # 14     737
# # 21     346
# # 26     232
# # 32     147
# # 47      87
# # 55      73
# # 63      52
# # 77      25

# # number of val samples per class: 
# # 2     1141
# # 3      810
# # 4      514
# # 7      292
# # 10     287
# # 9      242
# # 8      211
# # 14      97
# # 26      31
# # 21      21
# # 47      20
# # 32       7
# # 63       5
# # 55       3



# ######### creat csv file for action 
# noun_mapping_path = "/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_noun_classes.csv"
# verb_mapping_path = "/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_verb_classes.csv"
# noun_mapping_label = pd.read_csv(noun_mapping_path)
# verb_mapping_label = pd.read_csv(verb_mapping_path)

# invesrse_noun_mapping_label = {}
# for k, v in noun_mapping_label['key'].items():
#     for item in eval(noun_mapping_label['instances'].values[k]):
#         invesrse_noun_mapping_label[item.strip()] = v.strip()


# invesrse_verb_mapping_label = {}
# for k, v in verb_mapping_label['key'].items():
#     for item in eval(verb_mapping_label['instances'].values[k]):
#         invesrse_verb_mapping_label[item.strip()] = v.strip()

    
# action_list = []
# train_label = pd.read_csv(root_add_train)
# for i in range (0,len(train_label)):
#     label_noun_name = train_label ['noun'][i]
#     label_verb_name = train_label ['verb'][i]
#     #find the row index of the noun in the noun mapping label
#     label_noun_name = invesrse_noun_mapping_label[label_noun_name]
#     label_verb_name = invesrse_verb_mapping_label[label_verb_name]
#     action_list.append(f"{label_verb_name}"+"_"+f"{label_noun_name}")
    

# val_label = pd.read_csv(root_add_val)
# for i in range (0,len(val_label)):
#     label_noun_name = val_label ['noun'][i]
#     label_verb_name = val_label ['verb'][i]
#     #find the row index of the noun in the noun mapping label
#     label_noun_name = invesrse_noun_mapping_label[label_noun_name]
#     label_verb_name = invesrse_verb_mapping_label[label_verb_name]
#     action_list.append(f"{label_verb_name}"+"_"+f"{label_noun_name}")

# # action_list = list(set(action_list)) 

# unique_action_list = []
# for item in action_list: 
#     if item not in unique_action_list: 
#         unique_action_list.append(item) 

# print(len(unique_action_list))
# action_mapping = {unique_action_list[i]:i for i in range(len(unique_action_list))}

# ###
# train_df = {'path':[], 'label_name':[], 'label_num':[]}
# val_df = {'path':[], 'label_name':[], 'label_num':[]}


# for i in range (0,len(train_label)):
#     label_noun_name = train_label ['noun'][i]
#     label_verb_name = train_label ['verb'][i]

#     label_noun_name = invesrse_noun_mapping_label[label_noun_name]
#     label_verb_name = invesrse_verb_mapping_label[label_verb_name]
#     action_cls_name = f"{label_verb_name}"+"_"+f"{label_noun_name}"
#     action_cls_num = action_mapping[action_cls_name]
#     path = os.path.join(video_mp4_root_add_train, f"video_{i}.mp4")
#     train_df['path'].append(path)
#     train_df['label_name'].append(action_cls_name)
#     train_df['label_num'].append(action_cls_num)
    
# for i in range (0,len(val_label)):
#     label_noun_name = val_label ['noun'][i]
#     label_verb_name = val_label ['verb'][i]

#     label_noun_name = invesrse_noun_mapping_label[label_noun_name]
#     label_verb_name = invesrse_verb_mapping_label[label_verb_name]
#     action_cls_name = f"{label_verb_name}"+"_"+f"{label_noun_name}"
#     action_cls_num = action_mapping[action_cls_name]
#     path = os.path.join(video_mp4_root_add_val, f"video_{i}.mp4")
#     val_df['path'].append(path)
#     val_df['label_name'].append(action_cls_name)
#     val_df['label_num'].append(action_cls_num)

# train_df = pd.DataFrame(train_df)
# val_df = pd.DataFrame(val_df)


# # to_csv() 

# csv_annotation_root = "/home/mona/SSVLI/dataset/epic_kitchens/annotation/action"
# if not os.path.exists(csv_annotation_root):
#     os.makedirs(csv_annotation_root)
# train_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "train.csv"), sep=' ', na_rep='', float_format=None, 
# columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
# compression='infer', quoting=None, quotechar='"', 
# chunksize=None, date_format=None, doublequote=True, escapechar=None, 
# decimal='.', errors='strict', storage_options=None)

# val_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "val.csv"), sep=' ', na_rep='', float_format=None, 
# columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
# compression='infer', quoting=None, quotechar='"', 
# chunksize=None, date_format=None, doublequote=True, escapechar=None, 
# decimal='.', errors='strict', storage_options=None)

vn_list = []
mapping_vn2narration = {}
for f in [
    '/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train.csv',
    '/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_validation.csv',
]:
    csv_reader = csv.reader(open(f))
    _ = next(csv_reader)  # skip the header
    for row in csv_reader:
        vn = '{}:{}'.format(int(row[10]), int(row[12]))
        narration = row[8]
        if vn not in vn_list:
            vn_list.append(vn)
        if vn not in mapping_vn2narration:
            mapping_vn2narration[vn] = [narration]
        else:
            mapping_vn2narration[vn].append(narration)
        # mapping_vn2narration[vn] = [narration]
vn_list = sorted(vn_list)
print('# of action= {}'.format(len(vn_list)))
mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
print(labels[:5])