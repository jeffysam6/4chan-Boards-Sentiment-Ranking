import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
from torch.utils.data import TensorDataset


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