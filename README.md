# KhasiNMT
### This is English to Khasi and Khasi to English Neural Machine Translation System 

### Instruction

#### Install required Package

1. Install PyTorch with GPU Capability
2. Install Tensorflow
3. pip install transformers
4. pip install dill
5. pip install --user -U nltk
   
   import nltk
   
   nltk.download('punkt')
   
 #### Download Byte-Pair Encoding(BPE) Model for Tokenization and Vocabulary construction.
 Link: https://drive.google.com/file/d/1ZpEdxUPIvvg7xycZvuclf24Al0az_YxS/view?usp=share_link
 
 Place the file in PreTrainModel/KhasiRoberta Directory



\# config_training.yaml

\## Where the samples will be written
save_data: weights #None #weights  #Folder where the weights are stored or else set to None if you want to start training from beginning
initilise_embedding: True #Set to 1 if Source/Target Embedding is to be initialise with context embedding
context_embedding : True

freeze_embedding: True # Set to False if you force the embedding not to learnt [Generally after a specific epoch we will enable embedding to learnt]
is_augment_data: True
\# Corpus opts:
data:
    train:
        path_src: data/New/Train_kha.txt
        path_tgt: data/New/Train_eng.txt
        #Augmented datasets are optional
        path_src_aug: data/New/Train_kha_aug.txt    # if is_augment_data is set to False then path_src_aug = None
        path_tgt_aug: data/New/Train_eng_aug.txt    # if is_augment_data is set to False then path_tgt_aug = None
    prefix:
        path_src: data/model_data/nonbreaking_prefix.kha
        path_tgt: data/model_data/nonbreaking_prefix.en
    pre_train_model:
        model_path: PreTrainModel/KhasiRoberta



language:
    src: kha
    trg: en

# HyperParameters
num_epochs: 1000
save_checkpoint: 45 #Save after 15 minutes
batchsize: [128] #[8192] #[8192] #[2048] #[8, 64, 128, 512,1024, 2048]
printevery: 100
learning_rate: [0.0001] #[0.1, 0.001, 0.0001, 0.00001, 0.000001]
shuffle: [False] #[True]  #[False, True]   if batch_sampler is use then shuffle = False
max_sentence_len: 160
val_frac: 0.10  # precentage data in validation
freq_threshold: 0 #freq_threshold for vacabulary
num_workers: 2 # No. of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
# set freeze_embedding: True for debugging
debugging: 0 #Set to 1 if we want to check the performance of different lr, batch_size, shuffle else 0
current_checkpoint: weights/current_checkpoint_en.pt
best_model: weights/best_model_en.pt

#Model Parameters [Avoid changing these parameters unless necessary]
embedded_dim: 768
num_layer: 6
number_head: 8
dropout: 0.1
forward_expansion: 4
SGDR: 1 #If True
floyd: 0 #If False
