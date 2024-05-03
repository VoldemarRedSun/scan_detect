from sklearn.metrics import roc_auc_score
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import os
from catboost import  Pool
import pandas as pd
from torch.utils.data import DataLoader, Dataset
messages = pd.read_csv('train_spam.csv')
import seaborn as sns
import matplotlib.pyplot as plt
# sns.countplot(x='text_type', data=messages)
# plt.show()

import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def fit_model(X_train,y_train):
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB(alpha=0.8)
    mnb.fit(X_train,y_train)
    # print(mnb.predict(X_test))





class CorpusDataset(Dataset):
    def __init__(self, df, encoder):
        self.df = df
        self.texts = self.df['text'].values[:300]
        self.encoder = encoder
        self.labels = self.encoder.transform(self.df['text_type'].values)[:300]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.labels[idx], self.texts[idx]



import torch

from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "intfloat/e5-base-v2"

# initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()


class EmbedModel:

    def __init__(self, model, tokenizer, loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.loader = loader
        self.tokenizer = tokenizer
        self.model.eval()



    def _batch_embeddings(self, batch_texts):
        tokens = self.tokenizer(
            batch_texts, padding=True, max_length=512, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.model(**tokens)
        last_hidden = out.last_hidden_state.masked_fill(
            ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        embeds = last_hidden.sum(dim=1) / \
                         tokens["attention_mask"].sum(dim=1)[..., None]
        return embeds.cpu().numpy().tolist()

    def get_embeddings_labels(self):
        all_embs = []
        all_labels = []
        for labels, texts in self.loader:
            with torch.no_grad():
                embs = self._batch_embeddings(texts)
                # all_embs = np.concatenate((all_embs,embs), axis=0)
                # all_labels = np.concatenate((all_labels, labels), axis=0)
                all_embs.extend(embs)
                all_labels.extend(labels)
        return np.array(all_labels), np.array(all_embs)










def fit_catboost(train_data):
    import catboost
    from catboost import CatBoostClassifier

    # Создание модели
    model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6)
    model.fit(train_data, verbose=True)

    # Оценка модели
    # predictions = model.predict(data)
    return model

def fit_catboost(train_data, test_data ):
    from catboost import CatBoostClassifier
    CatBoostClassifier(eval_metric='AUC', verbose=100)
    model = CatBoostClassifier(eval_metric='AUC', iterations=100, learning_rate=0.1, depth=6)
    model.fit(train_data, eval_set = test_data, verbose=True)

    return model
def test_foo():
    df = pd.read_csv('train_spam.csv')
    train, test = train_test_split(df, test_size=0.20, random_state=0)
    encoder = LabelEncoder().fit(test['text_type'].values)
    train_dataset = CorpusDataset(train, encoder)
    test_dataset = CorpusDataset(test, encoder)
    loader_train = DataLoader(train_dataset, batch_size=100, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=100, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "intfloat/e5-base-v2"

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    labels_train, embed_train = EmbedModel(model, tokenizer, loader_train).get_embeddings_labels()
    labels_test, embed_test = EmbedModel(model, tokenizer, loader_test).get_embeddings_labels()
    train_data, test_data = Pool(embed_train, labels_train), Pool(embed_test, labels_test)
    cb = fit_catboost(train_data, test_data)
    cb.eval_metric(test_data, 'AUC')
    # predictions = model.predict_proba(X_test)
    # roc_auc_score(y_test, predictions[:, 1])
    print()

# def test_embed():



if __name__ == "__main__":
    test_foo()

    