import torch
import pickle

from transformers import AutoTokenizer, BertForSequenceClassification

class Model:
    def get_bert(self):
        model = BertForSequenceClassification.from_pretrained('NotebookWork/Data/bert-base-multilingual-cased')
        tokenizer = AutoTokenizer.from_pretrained('NotebookWork/Data/bert-base-multilingual-cased')

        model.load_state_dict(torch.load('Models/bert/bert_model_weights.pth', map_location=torch.device('cpu')))
        model.eval()

        return model, tokenizer

    def get_logreg_deftext(self):
        with open('Models/linear_models/logreg_def_text.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('Models/tf-idf/tfidf.plk', 'rb') as f:
            tfidf = pickle.load(f)

        return model, tfidf

