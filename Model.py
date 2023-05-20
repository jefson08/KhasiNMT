import torch
import torch.nn as nn
import torch.optim as optim
from Encoder_Decoder import Encoder, EncoderContext, Decoder
from utilities import *
import numpy as np


class Transformer(nn.Module):
    def __init__(self,
                 scr_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size,
                 num_layers,
                 forward_expansion,
                 heads,
                 dropout,
                 device,
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(scr_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               forward_expansion,
                               dropout,
                               device
                               )
        self.encodercontext = EncoderContext(scr_vocab_size,
                                             embed_size,
                                             num_layers,
                                             heads,
                                             forward_expansion,
                                             dropout,
                                             device
                                             )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg, src_embed=None):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        if src_embed is None:
            enc_src = self.encoder(src, src_mask)
        else:
            enc_src = self.encodercontext(src, src_mask, src_embed)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


def get_model(opt, src_vocab, trg_vocab, src_pad_idx, trg_pad_idx, lr, betas, eps, train=True, model_path=False):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(
        src_vocab,
        trg_vocab,
        src_pad_idx,
        trg_pad_idx,
        opt.d_model,
        opt.n_layers,
        opt.forward_expansion,
        opt.heads,
        opt.dropout,
        opt.device
    )

    if torch.cuda.is_available():
        model = model.to(opt.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=betas, eps=eps)

    epoch = 0
    train_loss_min = np.Inf

    if opt.load_weights is not None:
        # model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
        # load_checkpoint(torch.load("weights/translation_checkpoint.pth.tar"), model,  optimizer)

        # #Overwrite the previous/saved optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
        if(train):
            model, optimizer, epoch, train_loss_min = load_ckp(
                opt.current_checkpoint, model, optimizer)
            # Overwrite the previous stores lr with a new lr incase we change the lr
            optimizer.state_dict()['param_groups'][0]['lr'] = lr
        else:
            model, optimizer, epoch, train_loss_min = load_ckp(
                model_path, model, optimizer)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        epoch = 0  # Train for the first time
    return model, optimizer, epoch, train_loss_min
