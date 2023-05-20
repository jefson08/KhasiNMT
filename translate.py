import argparse
import time
import torch
from Model import get_model
from utilities import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Beam import beam_search
import nltk
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
import yaml
import unicodedata
from tqdm import tqdm
from itertools import groupby
from transformers import RobertaTokenizer, RobertaForMaskedLM
# nltk.download('punkt')

'''
def get_synonym(word, SRC):
    nltk.download('wordnet')
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0
'''


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def clean_input_sentence(sentence, opt):
    '''
    we want is to remove the points that are not '.' like a.m., we want to remove those '.'
    so that the algorithm understand that am is one word and not two sentences
    '''

    for prefix in opt.src_data_prefix:
        sentence = sentence.replace(prefix, prefix+"###")
    sentence = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".###", sentence)
    sentence = re.sub(r"\.###", "", sentence)
    sentence = unicodedata.normalize("NFKD", sentence)
    sentence = re.sub(r"  +", " ", sentence)
    # sentence = sentence.strip().split('\n')

    sentence = opt.src_tokenizer(sentence)['input_ids'][1:-1]
    return sentence


def translate_sentence(sentence, model, opt):

    model.eval()
    append_spicial = ""
    if sentence.split()[-1][-1] == '?' or sentence.split()[-1][-1] == '.' or sentence.split()[-1][-1] == '!':
        append_spicial = sentence.split()[-1][-1]
    # sentence = SRC.preprocess(sentence)
    indexed = clean_input_sentence(sentence, opt)

    sentence = Variable(torch.LongTensor([indexed]))
    sentence = sentence.to(opt.device)

    sentence = beam_search(sentence, model, opt)
    sentence = opt.trg_tokenizer.decode(sentence)
    if append_spicial != "" and sentence != "":
        if bool(re.match('^[a-zA-Z0-9]+$', sentence.split()[-1])):
            sentence = sentence+append_spicial
    sentence = multiple_replace(
        {' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)
    return sentence


def translate(opt, model):
    # s = """You! Are you Tom? I am Danny."""
    sentences = nltk.tokenize.sent_tokenize(opt.text)

    translated = []
    for sentence in sentences:
        translated.append(translate_sentence(sentence, model, opt))

    return (' '.join(translated))


def main():
    parser = argparse.ArgumentParser()
    # Comment this if config file is taken in statically
    # parser.add_argument('-config', required=True)
    opt = parser.parse_args()
    # Add  {opt.config} if argument is passed Else Add {"config_translation.yaml"}
    with open("config_translation.yaml") as f:
        try:
            dict = yaml.load(f, Loader=yaml.FullLoader)
            opt.load_weights = None if dict["save_data"] == "None" else dict["save_data"]
            opt.src_data_prefix = dict["data"]["preprocess"]["path_src"]
            opt.k = int(dict["beam"])
            opt.max_strlen = int(dict["max_sentence_len"])
            opt.d_model = int(dict["embedded_dim"])
            opt.n_layers = int(dict["num_layer"])
            opt.src_lang = dict["language"]["src"]
            opt.trg_lang = dict["language"]["trg"]
            opt.heads = int(dict["number_head"])
            opt.dropout = float(dict["dropout"])
            opt.floyd = True if int(dict["floyd"]) == 1 else False
            opt.forward_expansion = int(dict["forward_expansion"])
            opt.lr = float(dict["learning_rate"])
            # opt.random_mask = float(dict["random_mask"])
            opt.model_path = dict["model_path"]
            opt.kha_pretrain = dict["data"]["pre_train_model"]["model_path"]
            opt.context_embedding = int(dict["context_embedding"])

        except yaml.YAMLError as e:
            print(e)

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert opt.k > 0
    assert opt.max_strlen > 10
    kha_pretrain_path = opt.kha_pretrain
    opt.kha_pretrain = RobertaTokenizer.from_pretrained(
        opt.kha_pretrain, max_len=512)
    opt.eng_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    if opt.src_lang == 'kha':
        opt.src_tokenizer = opt.kha_pretrain
        opt.src_vocab = list(opt.src_tokenizer.get_vocab())[:]
    else:
        opt.src_tokenizer = opt.eng_tokenizer
        opt.src_vocab = list(opt.src_tokenizer.get_vocab())[:]

    if opt.trg_lang == 'kha':
        opt.trg_tokenizer = opt.kha_pretrain
        opt.trg_vocab = list(opt.trg_tokenizer.get_vocab())[:]
    else:
        opt.trg_tokenizer = opt.eng_tokenizer
        opt.trg_vocab = list(opt.trg_tokenizer.get_vocab())[:]

    if opt.context_embedding:
        if opt.src_lang == 'kha':
            opt.src_model = RobertaForMaskedLM.from_pretrained(
                kha_pretrain_path, output_hidden_states=True).to(opt.device)
        else:
            opt.src_model = RobertaForMaskedLM.from_pretrained(
                "roberta-base", output_hidden_states=True).to(opt.device)

        if opt.trg_lang == 'kha':
            opt.trg_model = RobertaForMaskedLM.from_pretrained(
                kha_pretrain_path, output_hidden_states=True).to(opt.device)
        else:
            opt.trg_model = RobertaForMaskedLM.from_pretrained(
                "roberta-base", output_hidden_states=True).to(opt.device)

    model, _, _, _ = get_model(opt, opt.src_tokenizer.vocab_size, opt.trg_tokenizer.vocab_size,
                               opt.src_tokenizer.get_vocab()['<pad>'], opt.trg_tokenizer.get_vocab()['<pad>'], opt.lr, (0.9, 0.98), 1e-9, False, opt.model_path)
    # opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.src_data_prefix is not None:
        try:
            opt.src_data_prefix = open(
                opt.src_data_prefix, encoding='utf-8').read()
        except:
            print("error: '" + opt.src_data_prefix + "' file not found")
            quit()
        '''
        Getting the non_breaking_prefixes as a clean list of words with a point at the end so it easier to use
        '''
        opt.src_data_prefix = opt.src_data_prefix.split('\n')
        opt.src_data_prefix = [' '+pref+'.' for pref in opt.src_data_prefix]

    while True:
        opt.text = input(
            "Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text == "q":
            break
        if opt.text == 'f':
            try:
                opt.text = input(
                    "Enter the File name in .txt format:\nNote File should be located in: data/ Directory:\n")
                opt.text = 'data/'+opt.text
                # opt.text='data/Test_eng.txt'
                text = open(
                    opt.text, encoding='utf-8').read().strip().split('\n')
                process_txt = []
                for t in text:
                    t = re.sub(r"  +", " ", t)  # Extra Space
                    t = t.strip()  # Remove Extra leading and trailing space
                    process_txt.append(t)
                f = open("data/out.txt", "w")
                i = 0
                for opt.text in tqdm(process_txt):
                    phrase = translate(opt, model)
                    f.write(phrase+"\n")
                    i = i+1
                f.close()
                continue
            except:
                print("error opening or reading text file")
                continue
        else:
            phrase = translate(opt, model)
            print('> ' + phrase + '\n')
            continue


if __name__ == '__main__':
    main()
