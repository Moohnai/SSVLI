import torch
import clip
from PIL import Image
import pandas as pd
import json
def text_encoding(text):
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
    #######epic_kitchens
    # # read csv file
    # csv_train_path = "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train.csv"
    # csv_val_path = "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_validation.csv"


    
    # #noun
    # #read csv file pandas
    # # df_train = pd.read_csv(csv_train_path)
    # # df_val = pd.read_csv(csv_val_path)

    # ## preprocess text
    # # encoded_text_train_noun = [text_encoding(text) for text in df_train['noun']]
    # # encoded_text_val_noun = [text_encoding(text) for text in df_val['noun']]

    # #verb
    # # encoded_text_train_verbs = [text_encoding(text) for text in df_train['verb']]
    # # encoded_text_val_verbs = [text_encoding(text) for text in df_val['verb']]

    # # save encoded text in pt file
    # # torch.save(encoded_text_train_noun, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_noun_text.pt")
    # # torch.save(encoded_text_val_noun, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_val_noun_text.pt")
    
    # # torch.save(encoded_text_train_verbs, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_verb_text.pt")
    # # torch.save(encoded_text_val_verbs, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_val_verb_text.pt")
    
    # #action
    # csv_action_path_train = "../../home/mona/SSVLI/dataset/epic_kitchens/annotation/action/train.csv"
    # csv_action_path_val = "../../home/mona/SSVLI/dataset/epic_kitchens/annotation/action/val.csv"

    # df_train = pd.read_csv(csv_action_path_train, header=None, delimiter=' ')
    # df_val = pd.read_csv(csv_action_path_val, header=None, delimiter=' ')

    # #read only the second column from df_train
    # #replace the "_" with " " in the text
    # df_train[1] = df_train[1]

    # encoded_text_train_action = [text_encoding(text) for text in df_train[1].str.replace("_", " ")]
    # encoded_text_val_action = [text_encoding(text) for text in df_val[1].str.replace("_", " ")]

    # torch.save(encoded_text_train_action, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_action_text.pt")
    # torch.save(encoded_text_val_action, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_val_action_text.pt")


    ####################ssv2
    #resd a json file
    ssv2_path = "../../home/mona/SSVLI/dataset/ssv2/annotation/train-val.json"
    encoded_text_list = []
    with open(ssv2_path) as ssv2_path:
        # for i in range (0, len(ssv2_path.readlines())):
        while ssv2_line := ssv2_path.readline():
            # ssv2_line = ssv2_path.readline()
            ssv2_line = ssv2_line.replace(",\n", "")

            ssv2_json = json.loads(ssv2_line)
    
        # encoded_text= [text_encoding(text) for text in ssv2_text[0]['label']]
            encoded_text= text_encoding(ssv2_json['label'])
            encoded_text_list.append(encoded_text)
    torch.save(encoded_text_list, "../../home/mona/SSVLI/dataset/ssv2/ssv2_text.pt")


