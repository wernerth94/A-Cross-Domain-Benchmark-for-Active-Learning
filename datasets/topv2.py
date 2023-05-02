import os
from os.path import exists
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from core.data import BaseDataset, VectorDataset, normalize, postprocess_svm_data
import requests, zipfile
import pickle
import nltk

class TopV2(BaseDataset):

    def __init__(self, cache_folder:str, config:dict, pool_rng, encoded:bool,
                 data_file="topv2_al.pt"):
        assert "domain" in config["dataset"], "You need to specify the sub-set (i.e. 'alarm')"
        assert not encoded, "TopV2 only supports un-encoded version (which uses GLOVE embeddings)"
        self.raw_zip_file = os.path.join(cache_folder, "optimal_al_data.zip")
        self.raw_unzipped_file = os.path.join(cache_folder, "optimal_al_data/intent_classification/data/TOP.pkl")
        self.raw_train_file = os.path.join(cache_folder, "topv2_train.txt")
        self.raw_test_file = os.path.join(cache_folder, "topv2_test.txt")
        fitting_mode = "from_scratch" if encoded else "finetuning"
        super().__init__(cache_folder, config, pool_rng, encoded,
                         data_file, fitting_mode)


    def _download_data(self, target_to_one_hot=True):
        zip_url = "https://www.ismll.uni-hildesheim.de/personen/twerner/optimal-al-data.zip"

        if not exists(self.raw_zip_file):
            r = requests.get(zip_url)
            # z = zipfile.ZipFile(io.BytesIO(r.content))
            with open(self.raw_zip_file, 'wb') as f:
                f.write(r.content)

        if not exists(self.raw_unzipped_file):
            z = zipfile.ZipFile(self.raw_zip_file)
            z.extract('intent_classification/data/TOP.pkl', os.path.join(self.cache_folder, "optimal_al_data"))

        print("Extracting and Embedding...")
        if exists(self.raw_unzipped_file):
            data = pickle.load(open(self.raw_unzipped_file, "rb"))
            all_points = data[self.config["dataset"]["domain"].lower()]["raw"]
            embeddings = data['GLOVE_EMBEDDING']
            x = [p[0] for p in all_points]
            x = self._embed_texts(x, embeddings)
            y = [p[1] for p in all_points]
            one_hot_y = np.zeros((len(y), max(y) + 1))
            one_hot_y[np.arange(len(y)), y] = 1
            all_ids = list(range(len(x)))
            self.pool_rng.shuffle(all_ids)
            train_ids, test_ids = all_ids[:-5000], all_ids[-5000:]
            self.x_train, self.y_train = x[train_ids], one_hot_y[train_ids]
            self.x_test, self.y_test = x[test_ids], one_hot_y[test_ids]
            self._convert_data_to_tensors()



    def _embed_texts(self, sentences:list, embedding_dict:dict):
        embeddings = np.array(list(embedding_dict.values()))
        num_embedding, emb_dim = embeddings.shape
        unk_idx = num_embedding
        pad_idx = num_embedding + 1
        word_embedding_data = torch.cat((torch.tensor(embeddings).float(),
                                         torch.randn(1, emb_dim) * 0.01,             # <unk>
                                         torch.zeros(1, emb_dim)                     # <pad>
                                        ), dim=0)
        word_embedding = nn.Embedding.from_pretrained(word_embedding_data, freeze=True,
                                                      padding_idx=pad_idx)

        vocabs = {v: i for i, v in enumerate(embedding_dict.keys())}
        def _tokenize(sent):
            words = nltk.word_tokenize(sent)
            tks = list([vocabs.get(w, len(vocabs)) for w in words])
            return tks, words

        list_of_tokens, list_of_words = zip(*map(_tokenize, sentences))
        lens = list(map(len, list_of_tokens))
        max_len = max(lens)
        list_of_embeddings = []
        for tks, words in zip(list_of_tokens, list_of_words):
            pad_len = max_len - len(tks)
            if pad_len == 0:
                word_embs = word_embedding(torch.tensor(tks))
            else:
                pad_tks = (torch.ones(pad_len) * pad_idx).long()
                cat_tks = torch.cat((torch.tensor(tks), pad_tks))
                word_embs = word_embedding(cat_tks)
            embs = word_embs
            list_of_embeddings.append(embs)
        embeddings_batch = torch.stack(list_of_embeddings, dim=0)
        embeddings_pack = pack_padded_sequence(embeddings_batch, torch.tensor(lens),
                                               batch_first=True, enforce_sorted=False)

        return embeddings_batch

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
    with open(f"../configs/topv2.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    dataset = TopV2("../../datasets", config, np.random.default_rng(0), encoded=False)
    model = dataset.get_classifier(model_rng=torch.Generator())
    test_point = dataset.x_train[0].unsqueeze(0)
    model(test_point)
