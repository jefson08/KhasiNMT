import torch
import pandas as pd
import re
import os
import dill as pickle
import unicodedata
from torch.utils.data import DataLoader
from Custom_dataset import Train_Dataset, MyCollate
import shutil
from pathlib import Path
from Batch_sampler import BucketBatchSampler
import numpy as np
from tqdm import tqdm


def read_data(opt):

    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data, encoding='utf-8').read()
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data, encoding='utf-8').read()
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

    if opt.src_data_prefix is not None:
        try:
            opt.src_data_prefix = open(
                opt.src_data_prefix, encoding='utf-8').read()
        except:
            print("error: '" + opt.src_data_prefix + "' file not found")
            quit()

    if opt.trg_data_prefix is not None:
        try:
            opt.trg_data_prefix = open(
                opt.trg_data_prefix, encoding='utf-8').read()
        except:
            print("error: '" + opt.trg_data_prefix + "' file not found")
            quit()


def clean_data(opt):
    '''
    Getting the non_breaking_prefixes as a clean list of words with a point at the end so it easier to use
    '''
    opt.src_data_prefix = opt.src_data_prefix.split('\n')
    opt.src_data_prefix = [' '+pref+'.' for pref in opt.src_data_prefix]
    opt.trg_data_prefix = opt.trg_data_prefix.split('\n')
    opt.trg_data_prefix = [' '+pref+'.' for pref in opt.trg_data_prefix]

    '''
    we want is to remove the points that are not '.' like a.m., we want to remove those '.'
    so that the algorithm understand that am is one word and not two sentences
    '''

    for prefix in opt.src_data_prefix:
        opt.src_data = opt.src_data.replace(prefix, prefix+"###")
    opt.src_data = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".###", opt.src_data)
    opt.src_data = re.sub(r"\.###", "", opt.src_data)
    opt.src_data = unicodedata.normalize("NFKD", opt.src_data)
    opt.src_data = re.sub(r"  +", " ", opt.src_data)
    opt.src_data = opt.src_data.strip().split('\n')

    for prefix in opt.trg_data_prefix:
        opt.trg_data = opt.trg_data.replace(prefix, prefix+"###")
    opt.trg_data = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".###", opt.trg_data)
    opt.trg_data = re.sub(r"\.###", "", opt.trg_data)
    opt.trg_data = unicodedata.normalize("NFKD", opt.trg_data)
    opt.trg_data = re.sub(r"  +", " ", opt.trg_data)
    opt.trg_data = opt.trg_data.strip().split('\n')


# Augment function
'''
def augment_text(embed_aug, df, src, trg, opt):
    aug1 = []
    aug2 = []
    for x, y in tqdm(zip(df[src], df[trg])):
        if(opt.src_lang == 'kha'):
            aug1.append(x)
        else:
            aug1.extend(embed_aug.augment(x))

        if(opt.trg_lang == 'kha'):
            aug2.append(y)
        else:
            aug2.extend(embed_aug.augment(y))
    df_aug = pd.DataFrame(list(zip(aug1, aug2)), columns=[
                          src, trg])
    return df_aug
'''


def split_dataset(opt):
    train_file = 'data/train.json'
    path_train = Path(train_file)
    val_file = 'data/validation.json'
    path_val = Path(val_file)
    train = None
    val = None

    if opt.load_weights is None:
        raw_data = {'src': [line for line in opt.src_data],
                    'trg': [line for line in opt.trg_data]}
        df = pd.DataFrame(raw_data, columns=["src", "trg"])
        df = df.dropna().drop_duplicates()

        # Remove sentence with length greater than the opt.max_strlen/
        mask = (df['src'].str.count(' ') < opt.max_strlen) & (
            df['trg'].str.count(' ') < opt.max_strlen)
        df = df.loc[mask]
        # Reset indexs
        df = df.reset_index(drop=True)
        # Remove Empty rows
        # Replace blanks by NaN
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        # Split for train and validation datsets
        # val_frac = 0.1  # precentage data in val
        val_split_idx = int(len(df)*opt.val_frac)  # index on which to split
        df_idx = list(range(len(df)))  # create a list of ints till len of data
        # get indexes for validation and train
        val_idx, train_idx = df_idx[:val_split_idx], df_idx[val_split_idx:]
        # create the sets
        train = df.iloc[train_idx].reset_index().drop('index', axis=1)
        val = df.iloc[val_idx].reset_index().drop('index', axis=1)

        if path_train.is_file() or path_val.is_file():
            print(f'The file {train_file} or  {val_file} exists')
            os.remove(train_file)
            os.remove(val_file)
            train.to_json(train_file, orient='split',
                          compression='infer', index='false')
            val.to_json(val_file, orient='split',
                        compression='infer', index='false')
        else:
            train.to_json(train_file, orient='split',
                          compression='infer', index='false')
            val.to_json(val_file, orient='split',
                        compression='infer', index='false')

        # create weights dir to save weights
        os.mkdir('./weights')

    else:
        try:
            print("loading presaved train and val datasets...")
            train = pd.read_json(
                train_file, orient='split', compression='infer')
            val = pd.read_json(val_file, orient='split',
                               compression='infer')

        except:
            print(
                f"error loading train and validation files, please ensure they are in {train_file} and {val_file}")
            quit()

    # Augment train data if enabled
    # This will augment at every new train loop
    '''
    if opt.augment_data:
        embed_aug = EmbeddingAugmenter(
            pct_words_to_swap=0.1, transformations_per_example=1)
        df_aug = augment_text(embed_aug, train, 'src', 'trg', opt)
        train = pd.concat([train, df_aug], axis=0)
        train = train.reset_index(drop=True)
        train = train.drop_duplicates()
        train = train.reset_index(drop=True)
    '''
    return train, val


def create_dataset(train, validate, opt, src_tokenizer, trg_tokenizer, batchsize, shuffle):
    train_dataset = Train_Dataset(train, src_tokenizer, trg_tokenizer)
    train_src = list()
    train_trg = list()

    [(train_src.append(train_dataset[x][0]), train_trg.append(train_dataset[x][1]))
     for x in range(len(train_dataset))]
    # train_trg = [train_dataset[x][1] for x in range(len(train_dataset))]
    bucket_batch_sampler_tr = BucketBatchSampler(
        batchsize, train_src, train_trg)

    validate_dataset = Train_Dataset(validate, src_tokenizer, trg_tokenizer)
    validate_src=list()
    validate_trg=list()
    [(validate_src.append(validate_dataset[x][0]), validate_trg.append(validate_dataset[x][1]))
                    for x in range(len(validate_dataset))]
    # validate_trg = [validate_dataset[x][1]
    #                 for x in range(len(validate_dataset))]
    bucket_batch_sampler_val = BucketBatchSampler(
        batchsize, validate_src, validate_trg)
    opt.src_pad = src_tokenizer.get_vocab()['<pad>']
    opt.trg_pad = trg_tokenizer.get_vocab()['<pad>']
    train_iter = DataLoader(train_dataset, batch_sampler=bucket_batch_sampler_tr, num_workers=opt.num_workers,
                            shuffle=shuffle, pin_memory=False, collate_fn=MyCollate(opt.src_pad, opt.trg_pad))
    validate_iter = DataLoader(validate_dataset, batch_sampler=bucket_batch_sampler_val, num_workers=opt.num_workers,
                               shuffle=shuffle, pin_memory=False, collate_fn=MyCollate(opt.src_pad, opt.trg_pad))
    #opt.train_len = get_len(train_iter)
    opt.train_len = len(train_iter)
    if opt.train_len == 0:
        opt.train_len = 1

    return train_iter, validate_iter


def get_len(train):
    for i, x in enumerate(train):
        # print(i)
        pass
    return i


# def save_checkpoint(state, filename, msg):
#     print("=> Saving checkpoint ", msg)
#     torch.save(state, filename)


# def load_checkpoint(checkpoint, model, optimizer):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize train_loss_min from checkpoint to train_loss_min
    train_loss_min = checkpoint['train_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], train_loss_min


def create_embedding_matrix(vocab, embedding_dict, dimension):
    embedding_matrix = np.zeros((len(vocab.get_token), dimension))

    for word, index in vocab.stoi.items():
        if word in embedding_dict.itos:
            embedding_matrix[index] = embedding_dict.vectors[embedding_dict.stoi[word]]
    return embedding_matrix
