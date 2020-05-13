import torch
import pandas as pd
from sklearn.model_selection import train_test_split

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

print(df.groupby(['category','label','data_type']).count()) #summary
