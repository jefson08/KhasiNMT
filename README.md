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

#### Training Khasi to English Translation System with following parameters:
      Initialisation of Embedding layer with Pre-trained Embedding Weights
      Data Augmentation
      Appending Pre-trained Context encodiing to local learned encoding
      
1. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: https://drive.google.com/file/d/1ISJfoUho3T_ojzggH_lrg8VfD7DhkLx9/view?usp=share_link

3. Require File: (Store file in data/New Directory)

        path_src: data/New/Train_kha.txt
        
        path_tgt: data/New/Train_eng.txt
        
        #Augmented datasets are optional
        
        path_src_aug: data/New/Train_kha_aug.txt    # if is_augment_data is set to False then path_src_aug = None
        
        path_tgt_aug: data/New/Train_eng_aug.txt    # if is_augment_data is set to False then path_tgt_aug = None
        
 4. Note if Augmented corpus is not available then rename Train_kha_aug.txt and Train_eng_aug.txt to Train_kha.txt and Train_eng.txt respectively.
 5. Train the model 

    python train.py
    
 6. Translate/ Test your model
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    Link: https://drive.google.com/file/d/1oEvFk37uXdr9gGQMbGUoPPQjYtc1JGWR/view?usp=share_link
    
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory

    
