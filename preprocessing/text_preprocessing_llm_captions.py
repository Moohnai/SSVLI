import os
import torch
from tqdm import tqdm
import clip
from PIL import Image
import pandas as pd

def caption_encoding(text):
    """
    text: list of strings
    """
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-L/14", device=device)

    # read csv file
    csv_captions_path = "../../home/mona/SSVLI/preprocessing/ssv2_captions_train.csv"
    
    df_train = pd.read_csv(csv_captions_path)
    # get the basename of the path
    df_train["video"] = df_train["video"].apply(lambda x: os.path.basename(x))

    # sort the df_train dataframe based from the df_train_org dataframe
    # add a new column to df_train dataframe and get the video id from df_train["video"]
    df_train["video_id"] = df_train["video"].apply(lambda x: int(x.split("_")[1].split(".")[0]))
    # sort the df_train dataframe based on the video_id
    df_train = df_train.sort_values(by=["video_id"])
    # remove the video_id column
    df_train = df_train.drop(columns=["video_id"])
    # reset the index
    df_train = df_train.reset_index(drop=True)

    # maximum length of the caption
    max_caption_length = 77

    # encode the "video_captions" column
    encoded_video_captions_train = [caption_encoding(text[:max_caption_length]) for text in tqdm(df_train["video_caption"], total=len(df_train), desc="Encoding video captions")]
    # encode the "image_captions" column
    encoded_image_captions_train = [caption_encoding(text[:max_caption_length]) for text in tqdm(df_train["image_caption"], total=len(df_train), desc="Encoding image captions")]
    # encode the "mixed_captions" column
    encoded_mixed_captions_train = [caption_encoding(text[:max_caption_length]) for text in tqdm(df_train["mixed_caption"], total=len(df_train), desc="Encoding mixed captions")]

    torch.save(encoded_video_captions_train, "../../home/mona/SSVLI/dataset/ssv2/SSV2_train_video_captions.pt")
    torch.save(encoded_image_captions_train, "../../home/mona/SSVLI/dataset/ssv2/SSV2_train_image_captions.pt")
    torch.save(encoded_mixed_captions_train, "../../home/mona/SSVLI/dataset/ssv2/SSV2_train_mixed_captions.pt")

