import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class Train_Dataset(Dataset):
    '''
    Initiating Variables
    transform : If we want to add any augmentation
    '''

    def __init__(self, df, src_tokenizer, trg_tokenizer, transform=None):

        self.transform = transform
        self.df = df
        self.source_texts = df['src']
        self.target_texts = df['trg']
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def __len__(self):
        return len(self.df)

    '''
    __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
    target values using the vocabulary objects we created in __init__
    '''

    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]

        if self.transform is not None:
            source_text = self.transform(source_text)

        # numericalize texts ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        #numerialized_source = [self.source_vocab.stoi["<BOS>"]]
        #numerialized_source = []
        #numerialized_source += self.source_vocab.numericalize(source_text)
        # numerialized_source.append(self.source_vocab.stoi["<EOS>"])
        numerialized_source = self.src_tokenizer(source_text)[
            'input_ids'][1:-1]
        #numerialized_target = [self.target_vocab.stoi["<BOS>"]]
        #numerialized_target += self.target_vocab.numericalize(target_text)
        # numerialized_target.append(self.target_vocab.stoi["<EOS>"])
        numerialized_target = self.trg_tokenizer(target_text)['input_ids']
        # convert the list to tensor and return
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target)


'''
class to add padding to the batches
collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class
is used on single example
'''


class MyCollate:
    def __init__(self, src_pad_idx, trg_pad_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    # __call__: a default method
    # First the obj is created using MyCollate(pad_idx) in data loader
    # Then if obj(batch) is called -> __call__ runs by default

    def __call__(self, batch):
        # get all source indexed sentences of the batch
        source = [item[0] for item in batch]
        # pad them using pad_sequence method from pytorch.
        source = pad_sequence(source, batch_first=False,
                              padding_value=self.src_pad_idx)

        # get all target indexed sentences of the batch
        target = [item[1] for item in batch]
        # pad them using pad_sequence method from pytorch.
        target = pad_sequence(target, batch_first=False,
                              padding_value=self.trg_pad_idx)
        return source, target
