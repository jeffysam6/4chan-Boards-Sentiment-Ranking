import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer,BertForSequenceClassification,get_linear_schedule_with_warmup,AdamW

from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader,RandomSampler,SequentialSampler

import numpy as np

from sklearn.metrics import f1_score

import random
import tqdm

df = pd.read_csv("smile-dataset/smile-annotations-final.csv",
                    names = ['id','text','category'])

df.set_index('id',inplace=True)

# print(df.head()) first 5
# print(df.text.iloc[0]) column index
# print(df.category.value_counts()) count of each value

# df.drop( df[ df['category'] ==  "nocode" or "|" in df["category"] ].index , inplace=True)

df = df[df.category != 'nocode']

df = df[-df.category.str.contains('\|')]

# print(df.category.value_counts())

possible_labels = df.category.unique()

label_dict = {}

for i,label in enumerate(possible_labels):
    label_dict[label] = i

df["label"] = df.category.replace(label_dict)

# print(df.head())

X_train,X_val,y_train,y_val = train_test_split(
                                df.index.values,
                                df.label.values,
                                test_size = 0.15,
                                random_state= 17,
                                stratify=df.label.values
)

df["data_type"] = "not_set"

# print(df.head())

df.loc[X_train,"data_type"] = 'train'


df.loc[X_val,"data_type"] = 'test'

# print(df.head())

# print(df.groupby(['category','label','data_type']).count()) #summary


tokenizer = BertTokenizer.from_pretrained(
                        'bert-base-uncased',
                        do_lower_case = True
)

encoded_data_train = tokenizer.batch_encode_plus(
                        df[df.data_type == 'train'].text.values, #text values 
                        add_special_tokens = True,                 #token to indicate sentences
                        return_attention_mask = True,               #returns position of masking
                        pad_to_max_length = True,                   #pads 0 to make size same
                        max_length = 256,                           #assumed max length of a tweet
                        return_tensors = 'pt'                           #pt for pytorch
)

encoded_data_val = tokenizer.batch_encode_plus(
                        df[df.data_type == 'test'].text.values, #text values 
                        add_special_tokens = True,                 #token to indicate sentences
                        return_attention_mask = True,               #returns position of masking
                        pad_to_max_length = True,                   #pads 0 to make size same
                        max_length = 256,                           #assumed max length of a tweet
                        return_tensors = 'pt'                           #pt for pytorch
)

# print(df.data_type.unique())
# print(encoded_data_val.keys())

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=="train"].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=="test"].label.values)

dataset_train = TensorDataset(input_ids_train,attention_masks_train,
                                labels_train)

dataset_val = TensorDataset(input_ids_val,attention_masks_val,
                                labels_val)

print(len(dataset_train),len(dataset_val))

model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = len(label_dict),
            output_attentions = False,
            output_hidden_states = False
)

dataloader_train = DataLoader(
                    dataset_train,
                    sampler = RandomSampler(dataset_train),
                    batch_size = 32
)

dataloader_val = DataLoader(
                    dataset_val,
                    sampler = RandomSampler(dataset_val),
                    batch_size= 32
)

optimizer = AdamW(
                model.parameters(),
                lr=1e-5,#2e-5 > 5e-5
                eps=1e-8
)

epochs = 10

scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(dataloader_train)*epochs
)


def f1_score_func(preds,labels):
    
    preds_flat = np.argmax(preds,axis=1).flatten()
    
    labels_flat = labels.flatten()
    
    return f1_score(labels_flat,preds_flat,average='weighted')


def accuracy_per_class(preds,labels):
    
    label_dict_inverse = {v:k for k,v in label_dict.items()}
    
    preds_flat = np.argmax(preds,axis=1).flatten()
    
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f"Class: {label_dict_inverse[label]}")
        print(f"Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n")
        


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch('cpu')

model.to(device)


for epoch in tqdm(range(1,epochs+ 1)):
    
    model.train()
    
    loss_train_total = 0
    
    progress_bar = tqdm(dataloader_train,
                        des='Epoch {:1d}'.format(epoch),
                        leave = False,
                        disable = False)
    
    for batch in progress_bar:
        
        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {
            'input_ids'     : batch[0],
            'attention_mask': batch[1],
            'labels'        : batch[2]
        }
        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        
        
        torch.nn_utils.clip_grad_norm_(model.parameters(),1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'traininig_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
    
    torch.save(model.state_dict(), f'Models/BERT_ft_epoch{epoch}.model')
    
    
    tqdm.write(f'\nEpoch {epoch}')
    
    # loss_train_avg = loss_train_total/len(dataloader)