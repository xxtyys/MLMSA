from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import torch
from torchvision import transforms

import os, re
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


def preprocess_data(dloc, img_transform=None, txt_transform=None, txt_processor=None, clip=None):
    # filename: list of ids for train or valid or test
    # dloc: a folder that store the files 
    
    imgs = {}
    txts = {}

    img_dloc = os.path.join(dloc, "images")
    for path in os.listdir(img_dloc):
        filepath = os.path.join(img_dloc, path)
        if path.endswith(".jpg"):
            img_id = os.path.splitext(os.path.basename(filepath))[0]
            img = Image.open(filepath).convert('RGB')
            if img_transform:
                img = img_transform(img)
            else:
                img = transforms.ToTensor()(img)
            imgs[img_id]=img

    txt_dloc = os.path.join(dloc, "texts")
    for path in os.listdir(txt_dloc):
        filepath = os.path.join(txt_dloc, path)        
        if path.endswith(".txt"):
            txt_id = os.path.splitext(os.path.basename(filepath))[0]
            text = open(filepath, 'r', encoding='utf-8', errors='ignore').read().strip().lower()
            if txt_transform:
                text = txt_transform(text, txt_processor)
            # print(text)=when you get excited to watch the grammys and you get told rhianna is performing .
            txt = clip.tokenize(text)
            txt = txt.squeeze()
            # p txt -> torch.Size([77])
            txts[txt_id] = txt
    return imgs, txts


class MMDataset(Dataset):
    def __init__(self, file_names, labels, labels_text, labels_img, img_data, txt_data):
        self.file_names = file_names
        self.labels = labels
        self.labels_text = labels_text
        self.labels_img = labels_img
        self.img_data = img_data
        self.txt_data = txt_data

        # self.imgs = imgs 

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names[idx])

        # imgs = {}
        # txts = {}
        # # get all filename in the folders --> all_filenames

        # for idx, fn in enumerate(all_filenames):
        #     img = Image.open(os.path.join(self.dloc, 'images', fname+'.jpg')).convert('RGB')
        #     text = open(os.path.join(self.dloc, 'texts', fname+'.txt'), 'r', encoding='utf-8', errors='ignore').read().strip().lower()
        #     imgs[idx] = img
        #     txts[idx] = text

        img = self.img_data[fname]
        text= self.txt_data[fname]
        label = self.labels[idx]
        label_text = self.labels_text[idx]
        label_img = self.labels_img[idx]

        
        return img, text, label, label_text, label_img


# class MMDataset(Dataset):
#     def __init__(self, file_names, labels, labels_text, labels_img, dloc, img_transform=None, txt_transform=None, txt_processor=None, clip=None):
#         self.file_names = file_names
#         self.labels = labels
#         self.labels_text = labels_text
#         self.labels_img = labels_img
#         self.dloc = dloc
#         self.img_transform = img_transform
#         self.txt_transform = txt_transform
#         self.txt_processor = txt_processor
#         self.clip=clip

#         # self.imgs = imgs 

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         fname = str(self.file_names[idx])

#         # imgs = {}
#         # txts = {}
#         # # get all filename in the folders --> all_filenames

#         # for idx, fn in enumerate(all_filenames):
#         #     img = Image.open(os.path.join(self.dloc, 'images', fname+'.jpg')).convert('RGB')
#         #     text = open(os.path.join(self.dloc, 'texts', fname+'.txt'), 'r', encoding='utf-8', errors='ignore').read().strip().lower()
#         #     imgs[idx] = img
#         #     txts[idx] = text

#         img = Image.open(os.path.join(self.dloc, 'images', fname+'.jpg')).convert('RGB')
#         text = open(os.path.join(self.dloc, 'texts', fname+'.txt'), 'r', encoding='utf-8', errors='ignore').read().strip().lower()
#         label = self.labels[idx]
#         label_text = self.labels_text[idx]
#         label_img = self.labels_img[idx]
        

#         if self.img_transform:
#             img = self.img_transform(img)
#         else:
#             img = transforms.ToTensor()(img)
        
#         if self.txt_transform:
#             text = self.txt_transform(text, self.txt_processor)
#         txt = self.clip.tokenize(text)
#         txt = txt.squeeze()

#         return img, txt, label, label_text, label_img


def get_text_processor(word_stats='twitter', htag=True):
    return TextPreProcessor(
            # terms that will be normalized , 'number','money', 'time','date', 'percent' removed from below list
            normalize=['url', 'email', 'phone', 'user'],
            # terms that will be annotated
            annotate={"hashtag","allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter=word_stats,

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector=word_stats,

            unpack_hashtags=htag,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=True,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )



def process_tweet(tweet, text_processor):

    proc_tweet = text_processor.pre_process_doc(tweet)

    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]

    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)




def get_bert_embeddings(tweet, model, tokenizer, device):
    # Split the sentence into tokens.
    input_ids = torch.tensor([tokenizer.encode(tweet, add_special_tokens=True)]).to(device)
    #res = tokenizer.batch_encode_plus(tweets, padding="longest")
    #input_ids = torch.tensor(res["input_ids"]).to(device)
    #attention_mask = res["attention_mask"].to(device)
    #last_out, pooled_out, encoded_layers = model(input_ids=input_ids,
                                                #attention_mask=attention_mask,
                                                # return_dict=False)
    
    # Predict hidden states features for each layer
    with torch.no_grad():
        try:
            last_out, pooled_out, encoded_layers = model(input_ids, return_dict=False)
        except:
            last_out, encoded_layers = model(input_ids, return_dict=False)


    # Calculate the average of all 22 token vectors.
    sent_emb_last = torch.mean(last_out[0], dim=0).cpu().numpy()

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.
    # For each token in the sentence...
    for token in token_embeddings:
    
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec.cpu().numpy())

    sent_word_catavg = np.mean(token_vecs_cat, axis=0)

    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec.cpu().numpy())

    sent_word_sumavg = np.mean(token_vecs_sum, axis=0)

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[-2][0]

    # Calculate the average of all 22 token vectors.
    sent_emb_2_last = torch.mean(token_vecs, dim=0).cpu().numpy()

    return sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last
