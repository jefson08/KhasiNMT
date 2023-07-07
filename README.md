# KhasiNMT
## English to Khasi and Khasi to English Neural Machine Translation System 

## Instruction

### Install the required Packages

1. Install PyTorch with GPU Capability
2. Install Tensorflow
3. pip install transformers
4. pip install dill
5. pip install --user -U nltk

   
 ### Download Byte-Pair Encoding(BPE) Model for Tokenization and Vocabulary construction.
 Link: <a href="https://drive.google.com/file/d/159W189thNM9YMAAwZUfTcij-3vC0qOkS/view?usp=sharing">Download FIle </a>
 
 Place the file in PreTrainModel/KhasiRoberta Directory

### A. Training Khasi to English Translation System with the following parameters:
      a. Initialization of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. Appending Pre-trained Context encoding to locally learned encoding
      
1. Download the config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: <a href="https://drive.google.com/file/d/1edSvNb1mLln7_5OxKmLNBMcxgQGD21Yq/view?usp=sharing"> Download File </a>
3. Require File: (Store file in data/New Directory)
   
        path_src_aug: data/New/Train_kha_aug.txt   
        
        path_tgt_aug: data/New/Train_eng_aug.txt    
        
 5. Train the model 

    python train.py
    
 7. Translate/Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: <a href="https://drive.google.com/file/d/1DBkJ7-d0DAUqg921HUvN7ralmhp6dzUW/view?usp=sharing">Download File</a>
    
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory
    
    
 ### B. Training Khasi to English Translation System with the following parameters:
      a. Initialization of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. No Appending of Pre-trained Context encoding to local learned encoding
      
1. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: <a href="https://drive.google.com/file/d/1QaUC1K201CXd78-G_CHCSNLRvsWc28im/view?usp=sharing">Download File </a>

3. Require File: (Store file in data/New Directory)

        path_src_aug: data/New/Train_kha_aug.txt   
        
        path_tgt_aug: data/New/Train_eng_aug.txt    
        
        
 4. Train the model 

    python train.py
    
 5. Translate/ Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: <a href="https://drive.google.com/file/d/1VwXMw7xI-hKzrSEM7dmvuWZYNZwk7c1U/view?usp=sharing">Download File</a>
    
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory

### C. Training Khasi to English Translation System with the following parameters:
      a. No initialization of Embedding layer with Pre-trained Embedding Weights
      b. No data Augmentation
      c. No appending Pre-trained Context encoding to locally learned encoding
      
1. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: <a href="https://drive.google.com/file/d/1RRct90AQfhnLr2TT3STSioiOObd9I28y/view?usp=sharing">Download File </a>

3. Require File: (Store file in data/New Directory)

        path_src: data/New/Train_kha.txt
        path_tgt: data/New/Train_eng.txt
        
 4. Train the model 

    python train.py
    
 5. Translate/ Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: <a href="https://drive.google.com/file/d/1qF0fhAE0BKV3DLVkg_33Ak7XBOxI0mn-/view?usp=sharing">Download File</a>
    
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory
    
### D. Training English to Khasi Translation System with the following parameters:
      a. Initialization of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. Appending Pre-trained Context encoding to locally learned encoding
      
1. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: <a href="https://drive.google.com/file/d/1SjFqTGJvWEkKo2Iycog-tp_1MNY-D0NC/view?usp=sharing">Download File </a>

3. Require File: (Store file in data/New Directory)
   
        path_src_aug: data/New/Train_eng_aug.txt   
        path_tgt_aug: data/New/Train_kha_aug.txt    
        
 5. Train the model 

    python train.py
    
 6. Translate/ Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: <a href="https://drive.google.com/file/d/1TQ30DTuInBhH4G3OqFGBHXcg6HExkRF4/view?usp=sharing">Download File</a>
    
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory
    
 
### E. Training English to Khasi Translation System with the following parameters:
      a. Initialisation of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. No appending of Pre-trained Context encoding to locally learned encoding
      
1. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: <a href="https://drive.google.com/file/d/14BgsNwhzBMQXqkZqId59quxOHCvt-sna/view?usp=sharing">Download File</a>

3. Require File: (Store file in data/New Directory)

        path_src_aug: data/New/Train_eng_aug.txt   
        path_tgt_aug: data/New/Train_kha_aug.txt
       
 5. Train the model 

    python train.py
    
 6. Translate/ Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: <a href="https://drive.google.com/file/d/1Ghuu0pfZ0pEuG1iFn1f3Fo4unS4D4rl2/view?usp=sharing">Download File</a>
    
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory


### F. Training English to Khasi Translation System the following parameters:
      a. No initialisation of Embedding layer with Pre-trained Embedding Weights
      b. No Data Augmentation
      c. No appending of Pre-trained Context encoding to locally learned encoding
      
1. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: <a href="https://drive.google.com/file/d/1WUa1BPAAslTDZ1no3KCNzexr4ghwCwzM/view?usp=sharing">Download File</a>

3. Require File: (Store file in data/New Directory)

        path_src: data/New/Train_eng.txt
        path_tgt: data/New/Train_kha.txt
       
 5. Train the model 

    python train.py
    
 6. Translate/Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: <a href="https://drive.google.com/file/d/1-UpVPA6oTgyKcTa8TVpx9mV9WquxLEpZ/view?usp=sharing">Download File</a>
    
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory

  
# Translate with the existing Trained Model
### A. Khasi to English with the following parameters:
      a. Initialization of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. Appending Pre-trained Context encoding to locally learned encoding
      
1. Download the model and place/store in weights Directory

   Link: <a href="https://drive.google.com/file/d/1fuaVtjMIMl97bZSKM-NMvd9VNNee0r7-/view?usp=sharing">Download File</a>
   
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
  
   Link: <a href="https://drive.google.com/file/d/1DBkJ7-d0DAUqg921HUvN7ralmhp6dzUW/view?usp=sharing">Download File</a>
3. Translate:
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory
    
    
### B. Khasi to English with the following parameters:
      a. Initialization of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. No Appending of Pre-trained Context encoding to local learned encoding
1. Download the model and place/store in weights Directory
   
   Link: <a href="https://drive.google.com/file/d/1a-BdUhNjORRAxW11w1mweCFyA_Rfse48/view?usp=sharing">Download File</a>
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: <a href="https://drive.google.com/file/d/1VwXMw7xI-hKzrSEM7dmvuWZYNZwk7c1U/view?usp=sharing">Download File</a>
3. Translate:
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory

### C. Khasi to English with the following parameters:
      a. No initialization of Embedding layer with Pre-trained Embedding Weights
      b. No data Augmentation
      c. No appending Pre-trained Context encoding to locally learned encoding
1. Download the model and place/store it in the weights Directory
   
   Link: <a href="https://drive.google.com/file/d/1vuOHHB4UOs0p15xHCN-zG7WxWnLZrUsZ/view?usp=sharing">Download File</a>
2. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: <a href="https://drive.google.com/file/d/1qF0fhAE0BKV3DLVkg_33Ak7XBOxI0mn-/view?usp=sharing">Download File</a>
3. Translate:
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory
    
    
### D. English to Khasi with the following parameters:
      a. Initialization of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. Appending Pre-trained Context encoding to locally learned encoding
1. Download the model and place/store in weights Directory
   
   Link: <a href="https://drive.google.com/file/d/1-3OHJEomo3AKepejOLze6T59DkAJphK0/view?usp=sharing">Download File</a>
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: <a href="https://drive.google.com/file/d/1TQ30DTuInBhH4G3OqFGBHXcg6HExkRF4/view?usp=sharing">Download File</a>
3. Translate:
    
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory
    
    
### E. English to Khasi with the following parameters:
      a. Initialisation of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. No appending of Pre-trained Context encoding to locally learned encoding
1. Download the model and place/store it in the weights Directory
   
   Link: <a href="https://drive.google.com/file/d/1geLrNfECsk7fsMU-d6tW54-Zw2KjtX27/view?usp=sharing">Download File</a>
2. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: <a href="https://drive.google.com/file/d/1Ghuu0pfZ0pEuG1iFn1f3Fo4unS4D4rl2/view?usp=sharing">Download File</a>
3. Translate:
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory


### F. English to Khasi with the following parameters:
      a. No initialization of Embedding layer with Pre-trained Embedding Weights
      b. No data Augmentation
      c. No appending of Pre-trained Context encoding to locally learned encoding
1. Download the model and place/store it in the weights Directory
   
   Link: <a href="https://drive.google.com/file/d/1HNzuoBeKIH1jGeF7hIlq6smevjvtpOrz/view?usp=sharing">Download File</a>
2. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: <a href="https://drive.google.com/file/d/1-UpVPA6oTgyKcTa8TVpx9mV9WquxLEpZ/view?usp=sharing">Download File</a>
3. Translate:
    Required File: Store the Test_file.txt in the data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generated in /data Directory
    
## A Hybrid system determines which translation model to adopt depending on the length of the input sentence.
### A. Translation of Khasi to English
1. Download the model and place/store it in the weights Directory (Ignored if already downloaded)
   
   Link: <a href="https://drive.google.com/file/d/1NRzSETJiSKa-3xedrdHR5AJtcESwZpNK/view?usp=sharing">Download File </a>
   
   Link: <a href="https://drive.google.com/file/d/1CYgCe_MKsk0jjdCy7TaGLkuFaI14SvPV/view?usp=sharing">Download File </a>
2. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: <a href="https://drive.google.com/file/d/1qrII2eNaRb9fY5MH2V7NS7M9Vayebvud/view?usp=sharing">Download File </a>
3. Translate:
 Required File: Store the Test_file.txt in data Directory

 python translate_hybrid.py

 Translated file (out.txt) will be generated in /data Directory
   
 ### B. Translation of English to Khasi
1. Download the model and place/store in the weights Directory (Ignored if already downloaded)
   
   Link: <a href="https://drive.google.com/file/d/1nqaDLJ1q8DDV73CbgAbJHiwxp2H6b92z/view?usp=sharing">Download File </a>
   
   Link: <a href="https://drive.google.com/file/d/17-JY4fJxC0kLnHUWF4hG2VgGoU4tOg4L/view?usp=sharing">Download File </a>
2. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: <a href="https://drive.google.com/file/d/1jD11gYuPZaqqYus_uo6iheuxaPwHh6bD/view?usp=sharing">Download File </a>
3. Translate:
 Required File: Store the Test_file.txt in data Directory

 python translate_hybrid.py

 Translated file (out.txt) will be generated in /data Directory
 
## BLEU Score 
perl multi-bleu.perl data/Reference.txt < data/predicted.txt

## Semantic Similarity
Open SemanticSimilarity.ipynb in Jupyter Notebook to Calculate Semantic Similarity
