import os
from os.path import exists
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from core.data import BaseDataset, to_one_hot
import requests, zipfile
import json
import nltk
from string import punctuation

class News(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file="news_al.pt"):
        assert not encoded, "News Categories only supports un-encoded version (which uses GLOVE embeddings)"
        self.raw_zip_file = os.path.join(cache_folder, "news_categories.zip")
        self.raw_unzipped_file = os.path.join(cache_folder, "news_categories/News_Category_Dataset_v3.json")
        self.raw_embedding_file = os.path.join(cache_folder, "news_categories/glove_embeddings.json")
        self.raw_train_file = os.path.join(cache_folder, "news_train.txt")
        self.raw_test_file = os.path.join(cache_folder, "news_test.txt")
        self.embedding_data_file = os.path.join(cache_folder, "news_al_embeddings.pt")
        super().__init__(cache_folder, config, pool_rng, encoded, data_file)


    def _download_data(self, target_to_one_hot=True, max_sentence_len=80):
        zip_url = "https://www.ismll.uni-hildesheim.de/personen/twerner/news_categories.zip"

        print("This download may take multiple minutes (108MB)")
        if not exists(self.raw_zip_file):
            r = requests.get(zip_url)
            with open(self.raw_zip_file, 'wb') as f:
                f.write(r.content)

        if not exists(self.raw_unzipped_file):
            z = zipfile.ZipFile(self.raw_zip_file)
            z.extract('News_Category_Dataset_v3.json', os.path.join(self.cache_folder, "news_categories"))
            z.extract('glove_embeddings.json', os.path.join(self.cache_folder, "news_categories"))

        # We are using the top 15 categories as indicated by Kaggle: https://www.kaggle.com/datasets/rmisra/news-category-dataset
        used_categories = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY", "PARENTING", "HEALTHY LIVING",
                           "QUEER VOICES", "FOOD & DRINK", "BUSINESS", "COMEDY", "SPORTS", "BLACK VOICES", "HOME & LIVING", "PARENTS"]

        print("Extracting and Processing...")
        if exists(self.raw_unzipped_file):
            with open(self.raw_unzipped_file, "r") as f:
                lines = f.readlines()
            sentences = []
            labels = []
            for line in lines:
                data = json.loads(line)
                if data["category"] in used_categories:
                    sentences.append(data["short_description"])
                    labels.append(data["category"])
            del lines
            label_df = pd.DataFrame(labels, columns=["y"])
            label_df["y"] = label_df["y"].astype("category")
            labels = label_df["y"].cat.codes
            labels = to_one_hot(labels.values)

            # Loading
            with open(self.raw_embedding_file, "r") as f:
                embedding_dict = json.load(f)
            vocabs = {v: i for i, v in enumerate(embedding_dict.keys())}

            tokenized_sentences = []
            cleaned_labels = []
            for i in tqdm.tqdm(range(len(sentences))):
                words = nltk.word_tokenize(sentences[i])
                words = [w for w in words if w not in punctuation]
                if len(words) > 0 and len(words) <= max_sentence_len:
                    tks = list([vocabs.get(w, len(vocabs)) for w in words])
                    tokenized_sentences.append(tks)
                    cleaned_labels.append(labels[i])
            cleaned_labels = torch.Tensor(cleaned_labels)
            del labels

            embeddings = np.array(list(embedding_dict.values())).astype(np.float16)
            num_embedding, emb_dim = embeddings.shape
            unk_idx = num_embedding
            pad_idx = num_embedding + 1
            word_embedding_data = torch.cat((torch.tensor(embeddings).float(),
                                             torch.randn(1, emb_dim) * 0.01,      # <unkown>
                                             torch.zeros(1, emb_dim)              # <padding>
                                            ), dim=0)
            torch.save(word_embedding_data, self.embedding_data_file)

            tokenized_sentences = self._pad_texts(tokenized_sentences, pad_idx)
            tokenized_sentences = torch.stack(tokenized_sentences)

            all_ids = list(range(len(tokenized_sentences)))
            self.pool_rng.shuffle(all_ids)
            train_ids, test_ids = all_ids[:-7000], all_ids[-7000:]
            self.x_train, self.y_train = tokenized_sentences[train_ids], cleaned_labels[train_ids]
            self.x_test, self.y_test = tokenized_sentences[test_ids], cleaned_labels[test_ids]
            # self._convert_data_to_tensors()


    def _pad_texts(self, tokenized_sentences:list, pad_idx:int):
        lens = list(map(len, tokenized_sentences))
        max_len = max(lens)
        for i, tks in enumerate(tokenized_sentences):
            pad_len = max_len - len(tks)
            if pad_len > 0:
                pad_tks = (torch.ones(pad_len) * pad_idx).long()
                cat_tks = torch.cat((torch.tensor(tks), pad_tks))
                tokenized_sentences[i] = cat_tks
            else:
                tokenized_sentences[i] = torch.Tensor(tokenized_sentences[i]).int()

        return tokenized_sentences

    def load_pretext_data(self)->tuple[Dataset, Dataset]:
        raise NotImplementedError

    def get_pretext_transforms(self, config:dict)->transforms.Compose:
        raise NotImplementedError

    def get_pretext_validation_transforms(self, config:dict)->transforms.Compose:
        raise NotImplementedError

    def get_meta_data(self) ->str:
        s = super().get_meta_data() + '\n'
        s += "Source: https://arxiv.org/pdf/2010.03546v1.pdf \n" \
             "Embeddings: GitHub: https://proceedings.mlr.press/v130/zhou21b.html \n" \
             "Embedding Model: Pretrained GLOVE \n" \
             "Classifier: BiLSTM"
        return s

if __name__ == '__main__':
    import yaml
    import numpy as np
    with open(f"../configs/news.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    dataset = News("../../datasets", config, np.random.default_rng(0), encoded=False)
    model = dataset.get_classifier(model_rng=torch.Generator())
    test_point = dataset.x_train[0].unsqueeze(0)
    model(test_point)
