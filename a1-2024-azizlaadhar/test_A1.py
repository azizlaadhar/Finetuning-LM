from torch.utils.data import Dataset
import datasets
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
from utils import isEnglish, lowerCase, replaceRare, isUnkSeq

def test_data_clean(dataset):
    if len(dataset)==97725:
        print("Test Passed ✅")
    else:
        print("Test Failed 🟥: length of dataset is wrong!")
        
def test_vocab_build(dataset,token_freq_dict):
    if len(dataset)==97536:
        print("Test1 Passed ✅")
    else:
        print("Test1 Failed 🟥: length of dataset is wrong!")
    
    if token_freq_dict['<unk>']==218741:
        print("Test2 Passed ✅")
    else:
        print("Test2 Failed 🟥: frequency of <unk> is wrong!")
    
def test_rnn_dataset(rnn_dataset):
    if len(rnn_dataset)==97536:
        print("Test1 Passed ✅")
    else:
        print("Test1 Failed 🟥: length of rnn_dataset is wrong!")
    
    if rnn_dataset.pad_idx==2139:
        print("Test2 Passed ✅")
    else:
        print("Test2 Failed 🟥: pad_idx is wrong!")

def test_dataloaders(train_dataloader, test_dataloader):
    if len(train_dataloader)==10973:
        print("Test1 Passed ✅")
    else:
        print("Test1 Failed 🟥: length of train_dataloader is wrong!")
    
    if len(test_dataloader)==1220:
        print("Test1 Passed ✅")
    else:
        print("Test1 Failed 🟥: length of test_dataloader is wrong!")
    
def test_lstm_scratch(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'The model has {num_params:,} trainable parameters')
    if num_params==15372833:
        print("Test Passed ✅")
    else:
        print("Test Failed 🟥: number of trainable parameters is wrong!")
  
def test_lstm_pretrained(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'The model has {num_params:,} trainable parameters')
    if num_params==10399533:
        print("Test Passed ✅")
    else:
        print("Test Failed 🟥: number of trainable parameters is wrong!")
