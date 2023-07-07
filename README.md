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

### Training Khasi to English Translation System with the following parameters:
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
    
    
 ### Training Khasi to English Translation System with the following parameters:
      a. Initialization of Embedding layer with Pre-trained Embedding Weights
      b. Data Augmentation
      c. No Appending of Pre-trained Context encoding to local learned encoding
      
1. Download the Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: <a href="https://drive.google.com/file/d/1QaUC1K201CXd78-G_CHCSNLRvsWc28im/view?usp=sharing">Download File </a>

3. Require File: (Store file in data/New Directory)

        path_src: data/New/Train_kha.txt
        
        path_tgt: data/New/Train_eng.txt
        
        
 4. Train the model 

    python train.py
    
 5. Translate/ Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: <a href="https://drive.google.com/file/d/1VwXMw7xI-hKzrSEM7dmvuWZYNZwk7c1U/view?usp=sharing">Download File</a>
    
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory

### Training English to Khasi Translation System with following parameters:
      Initialisation of Embedding layer with Pre-trained Embedding Weights
      Data Augmentation
      Appending Pre-trained Context encoding to local learned encoding
      
1. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: https://drive.google.com/file/d/1yuknar0ideF-EgiF_ivkeePDzpEElbCD/view?usp=share_link

3. Require File: (Store file in data/New Directory)

        path_src: data/New/Train_eng.txt
        path_tgt: data/New/Train_kha.txt
        #Augmented datasets are optional
        path_src_aug: data/New/Train_eng_aug.txt    # if is_augment_data is set to False then path_src_aug = None
        path_tgt_aug: data/New/Train_kha_aug.txt    # if is_augment_data is set to False then path_tgt_aug = None
        
 4. Note if Augmented corpus is not available then rename Train_eng_aug.txt and Train_kha_aug.txt to Train_eng.txt and Train_kha.txt respectively.
 5. Train the model 

    python train.py
    
 6. Translate/ Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: https://drive.google.com/file/d/1o8wL4BpvTBCO8fVi3OBHYjSnMMYuaSFd/view?usp=share_link
    
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory
    
    
 
### Training English to Khasi Translation System without the following parameters:
      Initialisation of Embedding layer with Pre-trained Embedding Weights
      Data Augmentation
      Appending Pre-trained Context encoding to local learned encoding
      
1. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
2. Link: https://drive.google.com/file/d/1n2XXleCWkqlk5sQVnxNq7-YEB7q1x1xe/view?usp=share_link

3. Require File: (Store file in data/New Directory)

        path_src: data/New/Train_eng.txt
        path_tgt: data/New/Train_kha.txt
       
 5. Train the model 

    python train.py
    
 6. Translate/ Test your model
 
    Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
    
    Link: https://drive.google.com/file/d/1j3TnnzMBpljP5dWjE_40l5Kq6Wisbiut/view?usp=share_link
    
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory
   
## Translate with the existing Trained Model
### Khasi to English with Parameters
1. Download the model and place/store in weights Directory

   Link: https://drive.google.com/file/d/162AODqoBZfP0LVBH0SBEitzL_e0ZOSQA/view?usp=share_link
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
  
   Link: https://drive.google.com/file/d/1oEvFk37uXdr9gGQMbGUoPPQjYtc1JGWR/view?usp=share_link
3. Translate:
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory
    
    
### Khasi to English without Parameters
1. Download the model and place/store in weights Directory
   
   Link: https://drive.google.com/file/d/1YpKUbjxG4Nw0jyOzAfWBDDWCHvfUBLuU/view?usp=share_link
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: https://drive.google.com/file/d/1BzJdJRzXxDKH__LhAnx5eJoli3Ua04yx/view?usp=share_link
3. Translate:
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory
    
    
### English to Khasi with Parameters
1. Download the model and place/store in weights Directory
   
   Link: https://drive.google.com/file/d/1TTI-TMpYjWMCV63Fnw-HhPP1uUl7OQw9/view?usp=share_link
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: https://drive.google.com/file/d/1o8wL4BpvTBCO8fVi3OBHYjSnMMYuaSFd/view?usp=share_link
3. Translate:
    
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory
    
    
### English to Khasi without Parameters
1. Download the model and place/store in weights Directory
   
   Link: https://drive.google.com/file/d/1-CeOQ0og717ZJ6LuiJSyNuPVnV5fbUSd/view?usp=share_link
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: https://drive.google.com/file/d/1j3TnnzMBpljP5dWjE_40l5Kq6Wisbiut/view?usp=share_link
3. Translate:
    Required File: Store the Test_file.txt in data Directory
    
    python translate.py
    
    Translated file (out.txt) will be generate in /data Directory
    
## A Hybrid system determine which translation model to adopt depending on the length of the input sentence.
### Translation of Khasi to English
1. Download the model and place/store in weights Directory (Ignored if already downloaded)
   
   Link: https://drive.google.com/file/d/1YpKUbjxG4Nw0jyOzAfWBDDWCHvfUBLuU/view?usp=share_link
   
   Link: https://drive.google.com/file/d/162AODqoBZfP0LVBH0SBEitzL_e0ZOSQA/view?usp=share_link
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: https://drive.google.com/file/d/1tb-oVwnQIaPSE8xxVHOkFHXkK5YPAGcJ/view?usp=share_link 
3. Translate:
 Required File: Store the Test_file.txt in data Directory

 python translate_hybrid.py

 Translated file (out.txt) will be generate in /data Directory
   
 ### Translation of English to Khasi
1. Download the model and place/store in weights Directory (Ignored if already downloaded)
   
   Link: https://drive.google.com/file/d/1-CeOQ0og717ZJ6LuiJSyNuPVnV5fbUSd/view?usp=share_link
   
   Link: https://drive.google.com/file/d/1TTI-TMpYjWMCV63Fnw-HhPP1uUl7OQw9/view?usp=share_link
2. Download Config File and Place/Replace in the root path (Here root directory is KhasiNMT)
   
   Link: https://drive.google.com/file/d/168ebXKy5dpKO4ZEHi-3WkBVsTmMgT8g5/view?usp=share_link 
3. Translate:
 Required File: Store the Test_file.txt in data Directory

 python translate_hybrid.py

 Translated file (out.txt) will be generate in /data Directory
 
## BLEU Score 
perl multi-bleu.perl data/Reference.txt < data/predicted.txt

## Semantic Similarity
Open SemanticSimilarity.ipynb in Jupyter Notebook to Calculate Semantic Similarity
