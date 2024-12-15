# CEMRI GAN
synthesis CEMRI using NCEMRI based on GAN models

step 1：make you data list.
(1) process your onw data,it's best that paired NCEMRI and CEMI data keep on same path,as person1/NCEMRI and /person1/CEMRI.
(2) make your own data list by data_process.py, include the division of training set, validation set and test set. 


step 2: train!
(1) Put your data list path in the data_processing.py. 
(2) set own model save path or learning rate .
(3) select the training model in train.py and set trianing
(4) evaluate the trained model.


