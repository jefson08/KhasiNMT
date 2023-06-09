import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math
import sys


def init_vars(src, model, opt, src_word_embed_4=None):
    init_tok = opt.trg_tokenizer.get_vocab()['<s>']
    #src_mask = (src != SRC.stoi['<PAD>']).unsqueeze(-2)
    src_mask = model.make_src_mask(src)
    if opt.context_embedding:
        e_output = model.encodercontext(src, src_mask, src_word_embed_4)
    else:
        e_output = model.encoder(src, src_mask)
    outputs = torch.LongTensor([[init_tok]])
    outputs = outputs.to(opt.device)

    #trg_mask = nopeak_mask(1, opt)
    # trg_mask = torch.max(model.create_padding_mask(
    #     outputs, opt.trg_tokenizer.get_vocab()['<pad>']), model.create_look_ahead_mask(outputs))
    trg_mask = model.make_trg_mask(outputs)
    # trg_mask=model.create_look_ahead_mask(outputs)
    # trg_word_embed_4[:, 0:1] embedding of the outputs
    out = model.decoder(outputs, e_output, src_mask, trg_mask)
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(opt.k)
    '''
    out=torch.rand(3,2,4)
    print(out)

    tensor([[[0.2377, 0.7356, 0.7970, 0.5229],
         [0.7864, 0.5257, 0.9376, 0.9255]],

        [[0.3359, 0.0788, 0.0241, 0.5803],
         [0.7100, 0.4628, 0.2560, 0.8759]],

        [[0.4064, 0.1074, 0.3182, 0.9646],
         [0.0283, 0.5976, 0.5116, 0.9787]]])

    print(out[:,-1])
    print(out[:,-1].shape)
    tensor([[0.7864, 0.5257, 0.9376, 0.9255],
        [0.7100, 0.4628, 0.2560, 0.8759],
        [0.0283, 0.5976, 0.5116, 0.9787]])
    torch.Size([3, 4])
    '''
    log_scores = torch.Tensor([math.log(prob) if prob > 0 else math.log(sys.float_info.min)
                              for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_strlen).long()
    outputs = outputs.to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, e_output.size(-2), e_output.size(-1))
    e_outputs = e_outputs.to(opt.device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):

    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor(
        [math.log(p) if p > 0 else math.log(sys.float_info.min) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    # log_probs = torch.Tensor(
    #     [math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    #row = k_ix // k
    row = torch.div(k_ix, k, rounding_mode='floor')
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores

# Initialise context embedding [word vectors in a sentence using Pre-Train Roberta model]


def initializeEmbedding(e_model, e_device, sentence, e_tokenizer, is_source):
    sent = [e_tokenizer.decode(toks) for toks in sentence]
    sent = e_tokenizer(sent, return_tensors="pt", padding=True).to(e_device)
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


def beam_search(src, model, opt):
    if opt.context_embedding:
        # last parameter is 1 if its a source sentence else 0
        src_word_embed_4, src_n = initializeEmbedding(
            opt.src_model, opt.device, src, opt.src_tokenizer, 1)
        src_mask = model.make_src_mask(src_n)
        outputs, e_outputs, log_scores = init_vars(
            src_n, model, opt, src_word_embed_4)
    else:
        outputs, e_outputs, log_scores = init_vars(src, model, opt)
        src_mask = model.make_src_mask(src)
    eos_tok = opt.trg_tokenizer.get_vocab()['</s>']
    #src_mask = (src != SRC.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_strlen):
        # trg_mask = torch.max(model.create_padding_mask(outputs[:, :i], opt.trg_tokenizer.get_vocab()[
        #                      '<pad>']), model.create_look_ahead_mask(outputs[:, :i]))
        trg_mask = model.make_trg_mask(outputs[:, :i])
        out = model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask)

        out = F.softmax(out, dim=-1)

        outputs, log_scores = k_best_outputs(
            outputs, out, log_scores, i, opt.k)

        # Occurrences of end symbols for all input sentences.
        ones = (outputs == eos_tok).nonzero()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            j = vec[0]
            if sentence_lengths[j] == 0:  # First end symbol has not been found yet
                sentence_lengths[j] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        #length = (outputs[0]==eos_tok).nonzero()[0]
        try:
            length = (outputs[0] == eos_tok).nonzero()[0]
        except:
            # If not found end of sentence
            return [tok for tok in outputs[0][1:len(src[0])].cpu().numpy()]

        return [tok for tok in outputs[0][1:length].cpu().numpy()]

    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]
        return [tok for tok in outputs[ind][1:length].cpu().numpy()]
