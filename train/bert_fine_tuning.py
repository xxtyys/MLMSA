import sys
import argparse
import random

import torch
from torchvision import models
import torch.optim as optim
import tqdm

import pickle
import numpy as np
import json

import csv

from torch.utils.tensorboard import SummaryWriter

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
import clip

#sys.path.insert(0, './feature_extraction/')
from feature_helper_bert import *
#sys.path.insert(0, './train/')
from train_helper import *


parser = argparse.ArgumentParser(description='Extract Image and CLIP Features')
parser.add_argument('--vtype', type=str, default='clip',
                    help='imagenet | places | emotion | clip')
parser.add_argument('--ttype', type=str, default='robertabase',
                    help='bertbase | robertabase | clip')        
parser.add_argument('--mvsa', type=str, default='single',
                    help='single | multiple')
parser.add_argument('--ht', action='store_true')
parser.add_argument('--train_batch_size', type=int, default=32)
# parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100,
                    help='50, 75, 100')
parser.add_argument('--clip_lr', type=float, default=2e-7,
                    help='1e-4, 5e-5, 2e-5, 2e-7')
parser.add_argument('--bert_lr', type=float, default=2e-5,
                   help='1e-4, 5e-5, 2e-5, 2e-7')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='1e-4, 5e-5, 2e-5')
parser.add_argument('--ftype', type=str, default='feats',
                    help='feats | logits')
parser.add_argument('--bert_feat_type', type=str, 
                    choices=['catavg', 'sumavg', '2last', 'last'], 
                    default='sumavg',
                    help='sumavg, 2last, last')
parser.add_argument('--norm', type=int, default=1,
                    help='0 | 1')
parser.add_argument('--split', type=int, default=1,
                    help='1-10')
parser.add_argument('--smooth', action='store_true')
parser.add_argument("--max_grad_norm", default=1.0, type=float)                  
parser.add_argument('--avg', action='store_true')  
# parser.add_argument('--modal', type=str, default='combine',
#                     help='image | text| combine')
# parser.add_argument('--single_label', action='store_true')
parser.add_argument('--use_label_embedding', action='store_true')
parser.add_argument('--use_label_embedding_loss', action='store_true')
parser.add_argument('--freeze_param', action='store_true')
parser.add_argument('--unuse_multi_labels', action='store_true')
parser.add_argument('--output_filename', type=str, default='output')
parser.add_argument('--alpha', type=float, default=1.0, help='1')
parser.add_argument('--beta', type=float, default=1.0, help='0.8|1.0|1.2')
parser.add_argument('--gamma', type=float, default=1.0, help='0.8|1.0|1.2')
parser.add_argument('--write_csv_path', type=str, default="result_feb")
parser.add_argument('--write_tensorboard', type=str, default='tensorboard')
# parser.add_argument('--freeze_clip', action='store_true')
# parser.add_argument('--freeze_bert', action='store_true')



args = parser.parse_args()
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


tensorboard_path = args.write_tensorboard
writer = SummaryWriter(f'{tensorboard_path}/tensorboard_{args.ttype}_{args.vtype}_freeze{args.freeze_param}_{args.mvsa}_ht{args.ht}_cliplr{args.clip_lr}_bertlr{args.bert_lr}_lr{args.lr}_split{args.split}_sm{args.smooth}_bs{args.train_batch_size}_mmlb{args.unuse_multi_labels}_le{args.use_label_embedding}_leloss{args.use_label_embedding_loss}')


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

    txt_processor = get_text_processor(htag=args.ht)
    txt_transform = process_tweet

    dloc = 'data/mvsa_%s/'%(args.mvsa)
    # tr_ids = pd.read_csv(dloc+'splits/train_%d.txt'%(args.split), header=None).to_numpy().flatten()
    # vl_ids = pd.read_csv(dloc+'splits/val_%d.txt'%(args.split), header=None).to_numpy().flatten()
    # te_ids = pd.read_csv(dloc+'splits/test_%d.txt'%(args.split), header=None).to_numpy().flatten()


    # get the idx of train/val/test set in valid data
    ids = {
        'train': pd.read_csv(dloc+'splits/train_%d.txt'%(args.split), header=None).to_numpy().flatten(),
        'val': pd.read_csv(dloc+'splits/val_%d.txt'%(args.split), header=None).to_numpy().flatten(),
        'test': pd.read_csv(dloc+'splits/test_%d.txt'%(args.split), header=None).to_numpy().flatten()
    }
    #get the valid ids,label,text,img
    pair_df = pd.read_csv(dloc+'valid_pairlist.txt', header=None)

    # #only text label
    # all_labels_text = pair_df[2].to_numpy().flatten() #use single modal label of text
    # #only image label
    # all_labels_img = pair_df[3].to_numpy().flatten() #use single modal label of image
    # #multimodal label
    # all_labels_mm = pair_df[1].to_numpy().flatten() #use multimadal label
    
    # #all label that are valid
    all_labels = {
        'mm': pair_df[1].to_numpy().flatten(),
        'text': pair_df[2].to_numpy().flatten(),
        'img': pair_df[3].to_numpy().flatten()
    }

    #split label to different set
    # for i in ['text','img','mm']:
    #     exec('lab_train_{} = all_labels_{}[tr_ids] '.format(i,i))
    #     exec('lab_val_{} = all_labels_{}[vl_ids]'.format(i,i))
    #     exec('lab_test_{} = all_labels_{}[te_ids]'.format(i,i))
    # print(lab_train_mm)
    # exit()

    # lab = {
    #     "mm": {'train': all_labels_mm[tr_ids]}
    # }
    lab = {}
    for i in ['mm','text','img']:
        lab[i] = {}
        for j in ['train', 'val', 'test']:
            lab[i][j] = all_labels[i][ids[j]]


    # #label split by idx
    # lab_train = all_labels[tr_ids]
    # lab_val = all_labels[vl_ids]
    # lab_test = all_labels[te_ids]
    all_tweet_ids = pair_df[0].to_numpy().flatten()
    tr_tweet_ids = all_tweet_ids[ids['train']]
    vl_tweet_ids = all_tweet_ids[ids['val']]
    te_tweet_ids = all_tweet_ids[ids['test']]

    # image
    # clip_model, img_preprocess = clip.load('ViT-B/32', device=device)
    clip_model, img_preprocess = clip.load('ViT-B/32', jit=False, device=device)
    # Note clip model use float16 to store weights.
    # standard models usually use float32
    # float() return a float32 version of a tensor.
    # pdb it use image_features.dtype() if it is float32 or float16
    # Without converting it into float32, will cause nan issue, see more in https://github.com/openai/CLIP/issues/144
    clip_model = clip_model.float()
    
    #text
    bert_type = {'bertbase': (BertModel,    BertTokenizer, 'bert-base-uncased'), 
                'robertabase': (RobertaModel,    RobertaTokenizer, 'roberta-base')}[args.ttype]
    bert_tokenizer = bert_type[1].from_pretrained(bert_type[2])
    bert_model = bert_type[0].from_pretrained(bert_type[2], output_hidden_states=True)
    bert_model.to(device)


    # TODO change parameters
    imgs_data, txts_data = preprocess_data2(dloc, device, img_transform=img_preprocess, txt_transform=txt_transform, txt_processor=txt_processor,clip=clip, bert_tokenizer= bert_tokenizer)

    tr_dataset = MMDataset(tr_tweet_ids, lab['mm']['train'], lab['text']['train'], lab['img']['train'], imgs_data, txts_data)
    vl_dataset = MMDataset(vl_tweet_ids, lab['mm']['val'], lab['text']['val'], lab['img']['val'], imgs_data, txts_data)
    te_dataset = MMDataset(te_tweet_ids, lab['mm']['test'], lab['text']['test'], lab['img']['test'], imgs_data, txts_data)
    tr_loader = DataLoader(tr_dataset, batch_size=args.train_batch_size, sampler=RandomSampler(tr_dataset))
    vl_loader = DataLoader(vl_dataset, batch_size=args.eval_batch_size, sampler=SequentialSampler(vl_dataset))
    te_loader = DataLoader(te_dataset, batch_size=args.eval_batch_size, sampler=SequentialSampler(te_dataset))

   
    label_embed = None
    label_embed_proj = None
    if args.use_label_embedding:
        # label_embed = nn.Embedding(3, 3)
        label_embed = nn.Embedding(3, 512)
        label_embed_proj = nn.Linear(512, 3)
        label_embed.to(device)
        label_embed_proj.to(device)

    if args.bert_feat_type == 'catavg':
        tdim = 3072
    else:
        tdim = 768
    vdim = clip_model.transformer.width
    param_init_norm_std = clip_model.transformer.width ** -0.5
    classify_model = MultiMLP_2Mod(vdim, tdim, 'combine', param_init_norm_std)
    # classify_model = MultiMLP_2Mod2(vdim, tdim, 'combine', param_init_norm_std)#1.13

    classify_model.to(device)
    classify_model_text = None
    classify_model_img = None
    if not args.unuse_multi_labels:
        classify_model_text = MultiMLP_2Mod(vdim, tdim, 'text', param_init_norm_std)
        classify_model_text.to(device)
        classify_model_img = MultiMLP_2Mod(vdim, tdim, 'img', param_init_norm_std)
        classify_model_img.to(device)
        # calweightloss = weightloss()
        # calweightloss.to(device)



    # params = list(clip_model.parameters()) + list(classify_model.parameters())
    # params_text = list(clip_model.parameters()) + list(classify_model_text.parameters())
    # params_img = list(clip_model.parameters()) + list(classify_model_img.parameters())
    #####fix the lr in clip pre-train model
    # for params in clip_model.parameters():
    #     params.requires_grad = False
    #####change the lr in different part
    # params1 = clip_model.parameters()
    # params2 = classify_model.parameters()
    # for p in params:
    #     print(type(p))
    #     exit()
    # optimizer1 = optim.Adam(params1, args.lr1)
    # optimizer2 = optim.Adam(params2, args.lr2)
    #optimizer = optim.Adam(params, args.lr)
    
    # parameter optimization
    active_params = []
    active_params.append({'params': classify_model.parameters()})
    if not args.unuse_multi_labels:
        active_params.append({'params': classify_model_text.parameters()})
        active_params.append({'params': classify_model_img.parameters()})
        # active_params.append({'params': calweightloss.parameters()})
    if args.use_label_embedding:
        active_params.append({'params': label_embed.parameters()})
        active_params.append({'params': label_embed_proj.parameters()})
    if not args.freeze_param:
        active_params.append({'params': clip_model.parameters(), 'lr': args.clip_lr})
        active_params.append({'params': bert_model.parameters(), 'lr': args.bert_lr})
    optimizer = optim.Adam(active_params, lr=args.lr)

            

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5,verbose=True, factor=0.1)
    best_test_acc, best_test_f1, best_epoch = train(
        tr_loader, vl_loader, te_loader, clip_model, bert_model, 
        bert_tokenizer, args.bert_feat_type, classify_model, 
        classify_model_text, classify_model_img, label_embed, label_embed_proj, 
        args, optimizer, device)
    print("best_test_acc:",best_test_acc)
    print("best_test_f1:",best_test_f1)
    print("best_epoch:",best_epoch)
    if args.avg:
        with open (f"{args.write_csv_path}/{args.alpha}-{args.beta}-{args.gamma}-{args.output_filename}_{args.vtype}_{args.ttype}_ft_{args.mvsa}_ht{args.ht}_sm{args.smooth}_freeze{args.freeze_param}_cliplr{args.clip_lr}_bertlr{args.bert_lr}_lr{args.lr}_unmmlb{args.unuse_multi_labels}_le{args.use_label_embedding}_leloss{args.use_label_embedding_loss}_1-10.csv","a") as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow([best_test_acc,best_test_f1])
    else:
        with open (f"{args.write_csv_path}/{args.alpha}-{args.beta}-{args.gamma}-{args.output_filename}_{args.vtype}_{args.ttype}_ft_{args.mvsa}_ht{args.ht}_sm{args.smooth}_freeze{args.freeze_param}_cliplr{args.clip_lr}_bertlr{args.bert_lr}_lr{args.lr}_unmmlb{args.unuse_multi_labels}_le{args.use_label_embedding}_leloss{args.use_label_embedding_loss}_split{args.split}.csv","a") as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow([best_test_acc,best_test_f1])

def evaluate(clip_model, bert_model, bert_tokenizer, bert_feat_type, classify_model, loader, label_embed, label_embed_proj, args, device):
    clip_model.eval()
    bert_model.eval()
    classify_model.eval()
    test_loss = 0.0
    test_loss_mb = 0.0
    all_preds = []
    all_labels = []

    use_label_embedding = (True if label_embed is not None and label_embed_proj is not None else False)
    if use_label_embedding:
        label_embed.eval()
        label_embed_proj.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Progress of evaluation"):
            # img_inps, txt_inps, labels, labels_text, labels_img = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            img_inps, txt_inps, labels, labels_text, labels_img = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            
            # TODO: fix for fine-tuning
            image_features = clip_model.encode_image(img_inps)
            # txt_inps = [torch.tensor([bert_tokenizer.encode(txt_r, add_special_tokens=True)]).to(device) for txt_r in txt_raw]
            text_features = torch.stack([get_bert_embeddings(txt_inp, bert_model, bert_feat_type, train=False).to(device) for txt_inp in txt_inps])

            if use_label_embedding:
                # labels_emb_final = label_embed(labels)
                # labels_emb_text = label_embed(labels_text)
                # labels_emb_img = label_embed(labels_img)
                # text_features += labels_emb_final
                # image_features += labels_emb_final
                pass

            # forward
            outputs = classify_model(image_features, text_features, 'combine')
            preds = torch.argmax(outputs.data, 1)

            # without label embeddings v0 + v1
            test_loss += cal_loss(outputs, labels, smoothing=args.smooth).item()

            # # with label embeddings 
            # if use_label_embedding:
            #     test_loss_mb += cal_label_embedding_loss(outputs, labels_emb_final).item()
            # labels = torch.argmax(labels_emb, 1) 
            if args.use_label_embedding_loss:
                # loss_mm_emb = cal_label_embedding_loss(outputs, labels_emb)
                # loss_text_emb = cal_label_embedding_loss(outputs_text, labels_text_emb)
                # loss_img_emb = cal_label_embedding_loss(outputs_img, labels_img_emb)
                # test_loss_mb += (loss_mm_emb + loss_text_emb + loss_img_emb)
                pass

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
        eval_loss = test_loss + test_loss_mb
        acc = metrics.accuracy_score(all_labels, all_preds)
        f1 = metrics.f1_score(all_labels, all_preds, average='weighted')
        

    return eval_loss/len(loader), acc, f1, all_preds, all_labels


def train(tr_loader, 
          vl_loader, 
          te_loader, 
          clip_model,
          bert_model, 
          bert_tokenizer, 
          bert_feat_type, 
          classify_model, 
          classify_model_text, 
          classify_model_img, 
          label_embed, 
          label_embed_proj,
          args,
          optimizer, 
          device
    ):
    best_val_loss=100
    best_val_acc=0.0
    best_acc=0.0
    best_f1=0.0
    best_epoch=100

    use_label_embedding = (True if label_embed is not None and label_embed_proj is not None else False)
    if use_label_embedding:
        label_embed.train()
        label_embed_proj.train()

    for epoch in tqdm.tqdm(range(1, args.epochs+1), desc="Progress of all epochs"):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        print('-' * 10)
        if args.freeze_param:
            freeze_model_params(clip_model)
            freeze_model_params(bert_model)
        else:
            clip_model.train()
            bert_model.train()
        classify_model.train()
        if not args.unuse_multi_labels:
            classify_model_text.train()
            classify_model_img.train()
            # calweightloss.train()
        running_loss = 0.0
        running_corrects = 0
        for batch in tqdm.tqdm(tr_loader, desc="Progress of one epoch"):
            # img_inps, txt_inps, labels, labels_text, labels_img = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            img_inps, txt_inps, labels, labels_text, labels_img, fname = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5]
            # p len(txt_raw) -> 32
            # for txt_r in txt_raw:
            # txt_inps = [torch.tensor([bert_tokenizer.encode(txt_r, add_special_tokens=True)]).to(device) for txt_r in txt_raw]
            # txt_inps -> list; p txt_inps[0].size() -> torch.Size([1, 20]); 32*torch.Size([1, 20])
            
            # TODO: fix for fine-tuning
            image_features = clip_model.encode_image(img_inps)
            # p txt_inps.size() -> torch.Size([32, 1, 100])
            text_features = torch.stack([get_bert_embeddings(txt_inp, bert_model, bert_feat_type, train=True).to(device) for txt_inp in txt_inps])


            if use_label_embedding:
                labels_emb_final = label_embed(labels)
                labels_emb_text = label_embed(labels_text)
                labels_emb_img = label_embed(labels_img)
                text_features += labels_emb_text
                image_features += labels_emb_img

            # forward
            # TODO check if self.model=='img' is equal to model=='image'
            outputs = classify_model(image_features, text_features, 'combine')
            if not args.unuse_multi_labels:
                outputs_text = classify_model_text(image_features, text_features, 'text')
                outputs_img = classify_model_img(image_features, text_features, 'image')

            _, preds = torch.max(outputs, 1)
            if not args.unuse_multi_labels:
                _, preds_text = torch.max(outputs_text, 1)
                _, preds_img = torch.max(outputs_img, 1)

            #loss = criterion(outputs, labels)
            # TODO try .to(device)
            # without label embedding v0
            # loss_mm = cal_loss(outputs, labels, smoothing=args.smooth)
            # loss_text = cal_loss(outputs_text, labels_text, smoothing=args.smooth)
            # loss_img = cal_loss(outputs_img, labels_img, smoothing=args.smooth)
            # loss = loss_mm + loss_text + loss_img

            # without label embedding v1
            # loss_mm = cal_loss(preds, labels, smoothing=args.smooth)
            # loss_text = cal_loss(preds_text, labels_text, smoothing=args.smooth)
            # loss_img = cal_loss(preds_img, labels_img, smoothing=args.smooth)
            # loss = loss_mm + loss_text + loss_img

            # TODO
            loss_mm = cal_loss(outputs, labels, smoothing=args.smooth)
            loss = loss_mm
            if not args.unuse_multi_labels:
                loss_text = cal_loss(outputs_text, labels_text, smoothing=args.smooth)
                loss_img = cal_loss(outputs_img, labels_img, smoothing=args.smooth)
                # loss_vector = torch.stack([loss, loss_text, loss_img])
                # loss = calweightloss(loss_vector)
                loss = loss + args.beta * loss_text + args.gamma * loss_img


            if args.use_label_embedding_loss:
                loss_mm_emb = cal_label_embedding_loss(outputs, labels_emb_final)
                loss_text_emb = cal_label_embedding_loss(outputs_text, labels_text_emb)
                loss_img_emb = cal_label_embedding_loss(outputs_img, labels_img_emb)
                loss += loss_mm_emb + loss_text_emb + loss_img_emb



            ######## example
            # final_loss = ml_loss + ult_loss + uli_loss
            # final_loss = (a * ml_loss) + (b * ult_loss) + (c * uli_loss)
            # a + b + c = 1.0
            # normalize  
            # final_loss.backward()

            ######## example

            loss.backward()
            #one_param = next(clip_model.parameters())
            #print((clip_model.parameters()))

            # Clip gradient in case they are too large.
            #params = list(clip_model.parameters()) + list(classify_model.parameters())
            #torch.nn.utils.clip_grad_norm_(
            #    active_params, args.max_grad_norm
            #)
            # backward + optimize
            # optimizer1.step()
            # optimizer2.step()
            optimizer.step()
            # to do for making scheduler a more popular one instead of the one in original code.
            #scheduler.step() # original code seems do not have this. Then the lr will not be changed gradually and is always kept at the inial value
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            optimizer.zero_grad()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
            
            
            #print("preds:",preds)
            #print("labels.data:",labels.data)
            #print('f1',f1)
        train_loss = running_loss / len(tr_loader)
        writer.add_scalar('training loss',train_loss,epoch)

        train_acc = running_corrects * 1.0 / (len(tr_loader.dataset))
        writer.add_scalar('training acc',train_acc,epoch)

        print('Training Loss: {:.6f} Acc: {:.2f}'.format(train_loss, 100.0 * train_acc))

        val_loss, val_acc, val_f1, _, _ = evaluate(clip_model, bert_model, bert_tokenizer, bert_feat_type, classify_model, vl_loader, label_embed, label_embed_proj, args, device)
        writer.add_scalar('validation loss',val_loss,epoch)
        writer.add_scalar('validation acc',val_acc,epoch)
        writer.add_scalar('validation f1',val_f1,epoch)


        test_loss, test_acc, test_f1, _, _ = evaluate(clip_model, bert_model, bert_tokenizer, bert_feat_type, classify_model, te_loader, label_embed, label_embed_proj, args, device)
        writer.add_scalar('testing loss',test_loss,epoch)
        writer.add_scalar('testing acc',test_acc,epoch)
        writer.add_scalar('testing f1',test_f1,epoch)

        # if val_loss <= best_val_loss:
        #     best_val_loss=val_loss
        #     best_acc = test_acc
        #     best_f1 = test_f1
        #     best_epoch = epoch
        if val_acc >= best_val_acc:
            best_val_acc=val_acc
            best_acc = test_acc
            best_f1 = test_f1
            best_epoch = epoch


    return best_acc, best_f1, best_epoch
        


main()
