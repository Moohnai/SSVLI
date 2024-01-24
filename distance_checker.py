import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


# train_lable_embed_path = '../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_action_text.pt'
# train_caption_embed_path = '../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_image_captions.pt'
# epic_lable_train_path = '../../home/mona/SSVLI/dataset/epic_kitchens/annotation/action/train.csv'
# epic_captions_train_path = '../../home/mona/SSVLI/dataset/epic_kitchens/epic_captions_train.csv'


train_lable_embed_path = '../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_action_text.pt'
train_caption_embed_path = '../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_image_captions.pt'
ssv2_lable_train_path = '../../home/mona/SSVLI/dataset/epic_kitchens/annotation/action/train.csv'
ssv2_captions_train_path = '../../home/mona/SSVLI/dataset/ssv2/ssv2_captions.csv'

train_lable_embed = torch.load(train_lable_embed_path)
train_caption_embed = torch.load(train_caption_embed_path)
ssv2_lable_train = pd.read_csv(ssv2_lable_train_path, header=None, delimiter=' ')
ssv2_captions_train = pd.read_csv(ssv2_captions_train_path)



# Euclidean distance 
def euclidean_distance(x,y):
    return torch.dist(x,y)

# Cosine distance
def cosine_distance(x,y):
    return torch.nn.functional.cosine_similarity(x,y)


def dot_product(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    return torch.dot(x,y)

caption_dict = {}
epic_captions_train = pd.read_csv(ssv2_captions_train_path)
for i in range(len(epic_captions_train)):
    video = epic_captions_train['video'][i].split("/")[-1].split(".")[-2]
    caption = epic_captions_train['mixed_caption'][i]
    caption_dict[video] = caption


lable_dict = {}
euclidean_distance_dict = {}
cosine_distance_dict = {}
dot_product_dict = {}
for i in range (0,len(train_lable_embed)):
    euclidean_distance_ = euclidean_distance(train_lable_embed[i], train_caption_embed[i]).item()
    cosine_distance_ = cosine_distance(train_lable_embed[i], train_caption_embed[i]).item()
    dot_product_ = dot_product(train_lable_embed[i], train_caption_embed[i]).item()
    video = ssv2_lable_train[0][i].split("/")[-1].split(".")[-2]
    lable = ssv2_lable_train[1][i].replace("_", " ")
    caption = caption_dict[video]

    #fill out dictionary
    lable_dict[video] = lable
    caption_dict[video] = caption

    euclidean_distance_dict[video] = euclidean_distance_
    cosine_distance_dict[video] = cosine_distance_
    dot_product_dict[video] = dot_product_ 

    print(video, " Label: ",lable, " Caption: ", caption, " Euclidean distance: ", euclidean_distance(train_lable_embed[i], train_caption_embed[i]), 
          " Cosine distance: ", cosine_distance(train_lable_embed[i], train_caption_embed[i]), 
          " Dot product: ", dot_product(train_lable_embed[i], train_caption_embed[i]))

#average distance
print("Average Euclidean distance: ", np.mean(list(euclidean_distance_dict.values())))
print("Average Cosine distance: ", np.mean(list(cosine_distance_dict.values())))
# plot the histogram
plt.hist(euclidean_distance_dict.values(), bins=100)
plt.title('Euclidean distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('euclidean_distance.png')

plt.figure()
plt.hist(cosine_distance_dict.values(), bins=100)
plt.title('Cosine distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('cosine_distance.png')

plt.figure()
plt.hist(dot_product_dict.values(), bins=100)
plt.title('Dot product')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('dot_product.png')

# print 10 closest and 10 farthest videos
sorted_cosine_distance = sorted(cosine_distance_dict.items(), key=lambda kv: kv[1])

# print 10 closest and 10 farthest videos'lables and captions
print("10 farthest text embedding based on Cosine distance: ")
for i in range(10):
    print("Video: ", sorted_cosine_distance[i][0], " Label: ", lable_dict[sorted_cosine_distance[i][0]], " Caption: ", caption_dict[sorted_cosine_distance[i][0]])

print("10 closest text embeddings based on Cosine distance: ")
for i in range(len(sorted_cosine_distance)-10, len(sorted_cosine_distance)):
    print("Video: ", sorted_cosine_distance[i][0], " Label: ", lable_dict[sorted_cosine_distance[i][0]], " Caption: ", caption_dict[sorted_cosine_distance[i][0]])








