Instruction to Train:
Type the below command in Terminal

export PYTHONPATH='/home/jefson08/2_Codes/Bert/BERT_From_Scratch/spaCy'
python train.py -config config_training.yaml


Instruction to Tranlate:
Type the below command in Terminal
python translate.py -config config_translation.yaml

##Error##
AttributeError: module 'torchtext.data' has no attribute 'Iterator'

#Change
from torchtext.legacy.data import Iterator

Instruction to Calculate Bleu Score:
Type the below command in Terminal
python Bleu.py -lang kha -reference data/khasi_test.txt -candidate data/out.txt

(Debugging)
python Bleu.py -lang kha -reference data/Debugging_Data/target.txt -candidate data/Debugging_Data/out.txt

Bleu Score: 0.47871288657188416



Bleu Score Using moses perl script
#With Phrase Table
perl multi-bleu.perl -lc Bleu_Score/with_phrase/khasi_test.txt < Bleu_Score/with_phrase/out.txt


#Without Phrase Table
perl multi-bleu.perl -lc Bleu_Score/w_o_phrase/khasi_test.txt < Bleu_Score/w_o_phrase/out.txt


perl multi-bleu.perl -lc data/khasi_test.txt < data/out.txt

python Bleu.py -lang kha -reference data/khasi_test.txt -candidate data/out.txt


