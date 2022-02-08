import torch
from transformers import BertTokenizer, BertForMaskedLM

torch.manual_seed(1)
build_big_vocab = 1

def total_vocab_builder():
    pass


def mlm_model(args, feature, label, id):
    input_ids = feature  # This is a unmasked tree
    labels = label  # This is a
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')



