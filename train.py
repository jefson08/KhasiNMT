from operator import truediv
import os
import secrets  # Debugging purpose
import shutil
import argparse
import time
import torch
import yaml
# from Models import get_model
from Model import get_model
from transformers import RobertaTokenizer, RobertaForMaskedLM
from utilities import *
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
from collections import namedtuple
from itertools import product

import pandas as pd
import json
from IPython.display import display, clear_output
from tqdm import tqdm
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


'''
RunManager class
'''


class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.epoch_num_correct = 0
        self.epoch_start_time = 0
        self.opt = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        # self.cptime = None

        self.network = None
        self.loader = None
        self.tb = None
        self.batch_size = 0

    def begin_run(self, run, network, opt, batch_size, src_model_emb):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = opt.train
        self.opt = opt
        self.batch_size = batch_size
        self.src_model_emb = src_model_emb
        # if opt.checkpoint > 0:
        #     self.cptime = time.time()
        self.tb = SummaryWriter(comment=f'-{run}')
        if self.opt.load_weights is None:
            '''
            self.tb.add_image('Images', grid)
            self.tb.add_graph(network, immages)
            '''
            for _, batch in enumerate(opt.train):
                src = batch[0].transpose(0, 1).to(opt.device).long()
                trg = batch[1].transpose(0, 1).to(opt.device).long()
                trg_input = trg[:, :-1]
                # preds = model(src, trg_input)
                # self.tb.add_graph(network, (src, trg_input))
                if self.opt.context_embedding:
                    src_word_embed_4, src = self.initializeEmbedding(
                        self.src_model_emb, self.opt.device, src, self.opt.src_tokenizer, 1)
                    self.tb.add_graph(
                        network, (src, trg_input, src_word_embed_4))
                else:
                    self.tb.add_graph(network, (src, trg_input))
                break

    '''
    Safe embedding in text file
    '''

    def write_embeddings(self, path, embeddings, vocab):
        with open(path, 'w') as f:
            for i, embedding in enumerate(tqdm(embeddings)):
                word = vocab[i]
                # skip words with unicode symbols
                if len(word) != len(word.encode()):
                    continue
                vector = ' '.join([str(i) for i in embedding.tolist()])
                f.write(f'{word} {vector}\n')

    def end_run(self, src_vocab, trg_vocab, batch_idx):
        self.tb.close()
        self.run_count = 0
        save_embeddings = self.network.encoder.word_embedding.weight.detach().cpu()
        self.write_embeddings(
            'weights/src_trained_embeddings.txt', save_embeddings, src_vocab)

        # Save Target embedding
        save_embeddings = self.network.decoder.word_embedding.weight.detach().cpu()
        # torch.save(save_embeddings, 'weights/trg_embedding_weights.pt')
        self.write_embeddings(
            'weights/trg_trained_embeddings.txt', save_embeddings, trg_vocab)

        enc_word_embedding = self.network.encoder.word_embedding.weight
        enc_words = src_vocab
        dec_word_embedding = self.network.decoder.word_embedding.weight
        dec_words = trg_vocab
        self.tb.add_embedding(enc_word_embedding, metadata=enc_words,
                              tag=f'enc_word embedding', global_step=batch_idx)
        self.tb.add_embedding(dec_word_embedding, metadata=dec_words,
                              tag=f'dec_word embedding', global_step=batch_idx)

        if self.opt.debugging == 1:
            try:
                # location
                location = os.getcwd()
                # directory
                dir = "weights"
                # path
                path = os.path.join(location, dir)
                shutil.rmtree(path)
                print("Directory '% s' has been removed successfully" % path)
            except OSError as error:
                print(error)
                print("Directory '% s' can not be removed" % path)

    def begin_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.epoch_count = epoch
        self.train_loss = 0.0
        self.valid_loss = 0.0

    def end_epoch(self, lr, batchsize, no_correct, total_words):
        epoch_duration = time.time()-self.epoch_start_time
        run_duration = time.time()-self.run_start_time
        t_loss = self.train_loss/len(self.loader.dataset)
        v_loss = self.valid_loss/len(self.loader.dataset)
        accuracy = float(no_correct)/float(total_words)
        self.tb.add_scalar('Training Loss', t_loss, self.epoch_count)
        self.tb.add_scalar('Validation Loss', v_loss, self.epoch_count)
        self.tb.add_scalar('Number correct', no_correct, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        if self.opt.freeze_embedding:
            if self.opt.debugging:
                for name, param in self.network.named_parameters():
                    self.tb.add_histogram(name, param, self.epoch_count)
                    self.tb.add_histogram(
                        f'{name}.grad', param.grad, self.epoch_count)

        self.tb.add_hparams({'lr': lr, 'batch_size': batchsize},
                            {'train_accuracy': accuracy, 'training loss': t_loss, 'validation loss': v_loss})
        results = OrderedDict()
        results['Run'] = self.run_count
        results['Epochs'] = self.epoch_count
        results['Training Loss'] = t_loss
        results['Validation Loss'] = v_loss
        results['Accuracy'] = accuracy
        results['Epoch Duration'] = epoch_duration
        results['Run Duration'] = run_duration

        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        clear_output(wait=True)
        display(df)

    def track_train_loss(self, loss, batch_idx):
        self.train_loss = self.train_loss + \
            ((1 / (batch_idx + 1)) * (loss - self.train_loss))

    def track_valid_loss(self, loss, batch_idx):
        self.valid_loss = self.valid_loss + \
            ((1 / (batch_idx + 1)) * (loss - self.valid_loss))

    def save(self, filename):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

# Initialise context embedding [word vectors in a sentence using Pre-Train Roberta model]

    def initializeEmbedding(self, e_model, e_device, sentence, e_tokenizer, is_source):
        sent = [e_tokenizer.decode(toks) for toks in sentence]
        sent = e_tokenizer(sent, return_tensors="pt",
                           padding=True).to(e_device)
        e_model.eval()
        # Run the text through Roberta, get the output and collect all of the hidden states produced
        # from all 6 layers.
        with torch.no_grad():
            outputs = e_model(**sent)
            # can use last hidden state as word embeddings
            hidden_states = outputs.hidden_states
            # initial embeddings can be taken from 0th layer of hidden states
            # word_embed_1 = hidden_states[0]
            # sum of all hidden states
            # word_embed_2 = torch.stack(hidden_states).sum(0)
            # sum of second to last layer
            # word_embed_3 = torch.stack(hidden_states[2:]).sum(0)
            # sum of last four layer
            word_embed_4 = torch.stack(hidden_states[-4:]).sum(0)
            # concat last four layers
            # word_embed_5 = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4]], dim=-1)

        # model parameter has to be passed
        # with torch.no_grad():
        #     for i, row in enumerate(sentence):
        #         for j, _ in enumerate(row):
        #             model.encoder.word_embedding.weight[sentence[i, j]
        #                                                 ] = word_embed_4[i, j]

        # Best pretrain embedding to be initiliase in the current train model
        # First token and last token embedding is ignored
        if is_source:
            # embedding, src
            return word_embed_4[:, 1:-1].to(e_device), sent['input_ids'][:, 1:-1].to(e_device)
        else:
            # embedding, trg_input, trg
            return word_embed_4[:, :-1].to(e_device), sent['input_ids'][:, :-1].to(e_device), sent['input_ids'].to(e_device)


def main():
    parser = argparse.ArgumentParser()
    # Comment this if config file is taken in statically
    # parser.add_argument('-config', required=True)
    opt = parser.parse_args()
    # Add  {opt.config} if argument is passed Else Add {"config_training.yaml"}
    with open("config_training.yaml") as f:
        try:
            dict = yaml.load(f, Loader=yaml.FullLoader)
            augment_data = int(dict["is_augment_data"])
            if augment_data:
                opt.src_data = dict["data"]["train"]["path_src_aug"]
                opt.trg_data = dict["data"]["train"]["path_tgt_aug"]
            else:
                opt.src_data = dict["data"]["train"]["path_src"]
                opt.trg_data = dict["data"]["train"]["path_tgt"]
            opt.src_data_prefix = dict["data"]["prefix"]["path_src"]
            opt.trg_data_prefix = dict["data"]["prefix"]["path_tgt"]
            opt.context_embedding = int(dict["context_embedding"])
            opt.load_pretrain_embed = int(dict["initilise_embedding"])
            opt.src_lang = dict["language"]["src"]
            opt.trg_lang = dict["language"]["trg"]
            opt.epochs = int(dict["num_epochs"])
            opt.d_model = int(dict["embedded_dim"])
            opt.n_layers = int(dict["num_layer"])
            opt.heads = int(dict["number_head"])
            opt.dropout = float(dict["dropout"])
            opt.batchsize = list(dict["batchsize"])
            opt.printevery = int(dict["printevery"])
            opt.lr = list(dict["learning_rate"])
            opt.shuffle = list(dict["shuffle"])
            opt.max_strlen = int(dict["max_sentence_len"])
            opt.load_weights = None if dict["save_data"] == "None" else dict["save_data"]
            opt.freeze_embedding = int(dict["freeze_embedding"])
            # Save after 15 minutes
            opt.save_checkpoint = int(dict["save_checkpoint"])
            opt.forward_expansion = int(dict["forward_expansion"])
            opt.SGDR = True if int(dict["SGDR"]) == 1 else False
            opt.floyd = True if int(dict["floyd"]) == 1 else False
            opt.debugging = int(dict["debugging"])
            opt.val_frac = float(dict["val_frac"])
            opt.freq_threshold = int(dict["freq_threshold"])
            opt.num_workers = int(dict["num_workers"])
            opt.current_checkpoint = dict["current_checkpoint"]
            opt.best_model = dict["best_model"]
            opt.kha_pretrain = dict["data"]["pre_train_model"]["model_path"]
        except yaml.YAMLError as e:
            print(e)

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    # path = os.getcwd()
    # print(path)
    src_model = None
    trg_model = None
    if opt.context_embedding or opt.load_pretrain_embed:
        if opt.src_lang == 'kha':
            src_model = RobertaForMaskedLM.from_pretrained(
                kha_pretrain_path, output_hidden_states=True).to(opt.device)
        else:
            src_model = RobertaForMaskedLM.from_pretrained(
                "roberta-base", output_hidden_states=True).to(opt.device)

        if opt.trg_lang == 'kha':
            trg_model = RobertaForMaskedLM.from_pretrained(
                kha_pretrain_path, output_hidden_states=True).to(opt.device)
        else:
            trg_model = RobertaForMaskedLM.from_pretrained(
                "roberta-base", output_hidden_states=True).to(opt.device)

    read_data(opt)
    clean_data(opt)
    train_ds, validate_ds = split_dataset(opt)
    params = OrderedDict(
        lr=opt.lr,
        batch_size=opt.batchsize,
        shuffle=opt.shuffle
    )
    m = RunManager()
    print("training model...")
    for run in RunBuilder.get_runs(params):
        print("Learning rate:{} Batch_size:{} Shuffle:{}".format(
            run.lr, run.batch_size, run.shuffle))
        opt.train, opt.validate = create_dataset(
            train_ds, validate_ds, opt, opt.src_tokenizer, opt.trg_tokenizer, run.batch_size, run.shuffle)
        model, opt.optimizer, start_epoch, train_loss_min = get_model(
            opt, opt.src_tokenizer.vocab_size, opt.trg_tokenizer.vocab_size, opt.src_pad, opt.trg_pad, run.lr, (0.9, 0.98), 1e-9)
        opt.criterion = nn.CrossEntropyLoss(
            ignore_index=opt.trg_pad)
        # opt.optimizer = torch.optim.Adam(model.parameters(), lr=run.lr, betas=(0.9, 0.98), eps=1e-9)

        '''
        To freeze the embedding weights, we set model.embedding.weight.requires_grad to False.
        This will cause no gradients to be calculated for the weights in the embedding layer,
        and thus no parameters will be updated when optimizer.step() is called.
        '''
        # Uncomment this below line if you want to freeze/unfreeze embedding layer
        if opt.freeze_embedding:
            model.encoder.word_embedding.requires_grad_(True)
            model.decoder.word_embedding.requires_grad_(True)
        else:
            model.encoder.word_embedding.requires_grad_(False)
            model.decoder.word_embedding.requires_grad_(False)

        if opt.SGDR == True:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

        # Uncomment below lines if you want to initialise pre-train embedding
        if opt.load_pretrain_embed and opt.load_weights is None:
            with torch.no_grad():
                model.encoder.word_embedding.weight[:
                                                    ] = src_model.roberta.embeddings.word_embeddings.weight[:]
                model.decoder.word_embedding.weight[:
                                                    ] = trg_model.roberta.embeddings.word_embeddings.weight[:]
        # if opt.checkpoint > 0:
        #     print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

        m.begin_run(run, model, opt, run.batch_size, src_model)
        print("Learning rate: %f" %
              (opt.optimizer.state_dict()['param_groups'][0]['lr']))

        start_time = time.time()
        time_count = 1

        for epoch in range(start_epoch, opt.epochs, 1):
            m.begin_epoch(epoch)
            total_seq = 0  # Represent to total no. of word in src/trg sentences
            num_correct = 0
            model.train()
            for i, batch in enumerate(opt.train):
                src = batch[0].transpose(0, 1).to(opt.device).long()
                trg = batch[1].transpose(0, 1).to(opt.device).long()
                trg_input = trg[:, :-1]

                opt.optimizer.zero_grad()
                if opt.context_embedding:
                    # last parameter is 1 if its a source sentence else 0
                    src_word_embed_4, src = m.initializeEmbedding(
                        src_model, opt.device, src, opt.src_tokenizer, 1)
                    # trg_word_embed_4, trg_input, trg = m.initializeEmbedding(
                    #     trg_model, opt.device, trg[:, 1:-1], opt.trg_tokenizer, 0)
                    # preds = model(src, trg_input,
                    #               src_word_embed_4, trg_word_embed_4)
                    preds = model(src, trg_input, src_word_embed_4)

                else:
                    preds = model(src, trg_input)
                ys = trg[:, 1:].contiguous().view(-1)
                loss = opt.criterion(preds.view(-1, preds.size(-1)), ys)
                # loss = torch.sum(loss+src_loss+trg_loss)
                loss.backward()
                opt.optimizer.step()
                if opt.SGDR == True:
                    opt.sched.step()
                m.track_train_loss(loss.item(), i)

            train_loss = m.train_loss / len(opt.train.dataset)

            # Save after specified time
            end_time = time.time()
            if opt.save_checkpoint > 0 and ((end_time - start_time)//60) // opt.save_checkpoint >= 1:
                print(
                    f'Saving model after {time_count * opt.save_checkpoint} minutes')
                # create checkpoint variable and add important data
                checkpoint = {
                    'epoch': epoch + 1,
                    'train_loss_min': train_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.optimizer.state_dict(),
                }

                # save checkpoint
                save_ckp(checkpoint, False,
                         opt.current_checkpoint, opt.best_model)
                # TODO: save the model if train loss has decreased
                if train_loss <= train_loss_min:
                    print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        train_loss_min, train_loss))
                    # save checkpoint as best model
                    save_ckp(checkpoint, True,
                             opt.current_checkpoint, opt.best_model)
                    train_loss_min = train_loss
                start_time = end_time
                time_count += 1
            # End Saving after specified time
            # Save after complete epoch
            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'train_loss_min': train_loss,
                'state_dict': model.state_dict(),
                'optimizer': opt.optimizer.state_dict(),
            }

            # save checkpoint
            save_ckp(checkpoint, False,
                     opt.current_checkpoint, opt.best_model)
            # TODO: save the model if train loss has decreased
            if train_loss <= train_loss_min:
                print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    train_loss_min, train_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True,
                         opt.current_checkpoint, opt.best_model)
                train_loss_min = train_loss
            # End saving after certain epoch
            model.eval()     # Optional when not using Model Specific layer
            with torch.no_grad():
                for i, batch in enumerate(opt.validate):
                    src = batch[0].transpose(0, 1).to(opt.device).long()
                    trg = batch[1].transpose(0, 1).to(opt.device).long()
                    trg_input = trg[:, :-1]

                    if opt.context_embedding:
                        # last parameter is 1 if its a source sentence else 0
                        src_word_embed_4, src = m.initializeEmbedding(
                            src_model, opt.device, src, opt.src_tokenizer, 1)
                        # trg_word_embed_4, trg_input, trg = m.initializeEmbedding(
                        #     trg_model, opt.device, trg[:, 1:-1], opt.trg_tokenizer, 0)
                        # preds = model(src, trg_input,
                        #               src_word_embed_4, trg_word_embed_4)
                        preds = model(src, trg_input, src_word_embed_4)
                    else:
                        preds = model(src, trg_input)
                    ys = trg[:, 1:].contiguous().view(-1)
                    loss = opt.criterion(preds.view(-1, preds.size(-1)), ys)
                    real_out = trg[:, 1:]
                    num_correct += (preds.max(2)[1] == real_out).sum()
                    total_seq += (real_out.shape[0]*real_out.shape[1])
                    m.track_valid_loss(loss.item(), i)
                    if (i + 1) % opt.printevery == 0:
                        p = int(100 * (i + 1) / opt.train_len)
                        t_loss = m.train_loss/opt.printevery
                        v_loss = m.valid_loss/opt.printevery
                        val_accuracy = float(num_correct)/float(total_seq)
                        print("   %dm: epoch %d [%s%s]  %d%%  train loss = %.3f  validation loss = %.3f  Validation_Accuracy= %.2f lr=%f" %
                              ((time.time() - m.run_start_time)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, t_loss, v_loss, val_accuracy, opt.optimizer.state_dict()['param_groups'][0]['lr']))

                        for row in preds.max(2)[1]:
                            # print([TRG.vocab.itos[i] for i in row])
                            pred_out = ' '.join([opt.trg_vocab[i.item()]
                                                for i in row[:-1]])
                        for row in real_out:
                            real = ' '.join([opt.trg_vocab[i.item()]
                                            for i in row[:-1]])

                        df = pd.DataFrame(
                            {'Predicted': [pred_out], 'Actual': [real]})
                        df.to_csv('Training_track.csv', mode='a', header=False)
            m.end_epoch(run.lr, run.batch_size, num_correct, total_seq)
        m.end_run(opt.src_vocab, opt.trg_vocab, i)
    m.save('results')


if __name__ == "__main__":
    main()
