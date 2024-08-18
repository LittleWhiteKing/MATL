# MATL
MATL: A deep neural network using multi-scale convolutions and transformer for transcription factor binding site prediction.

# Dependencies

python ==3.8

pytorch==1.10.2

numpy==1.21.5

pandas==1.3.5

scikit-learn==1.0.2

# Input
MATL takes two files as input: the Sequence file and the Shape file.The Sequence file is composed of two CSV files: one for training validation and one for testing. The datasets are available at http://cnn.csail.mit.edu/motif discovery/.The Shape file is computed from the corresponding DNA sequences in the Sequence file by the DNAshapeR tool, which can be downloaded from http://www.bioconductor.org/. The Shape file consists of ten CSV files of helix twist (HelT), minor groove width (MGW), propeller twist (ProT), rolling (Roll), and minor groove electrostatic potential (EP) for training validation data and testing data.

# Output
Run the `1_main_train.py` to train the data and get the results.  

# Finally
Thank you for taking your time to study my research. I hope my research can bring you some inspiration and insights, and I hope it can become a powerful tool. Finally, I wish you all the best

