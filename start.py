from usptoParser import USPTOParser
from dataCleaning import dataCleaning
from train import Tuned_SVM
# Step 1
#Read contained xml file
#files are downloaded from  https://bulkdata.uspto.gov/
# Parse the required information from xml like patent title,abstract,
#description,claims
USPTOParser.parseUSPTOXML()


#step 2
#clean the datset and save it into csv for model training
dataCleaning.processDATA()

#Step 3 
#Perfrom Exploratory data analysis

#Step 4 
#Evaluate different models and do hyperparameter tuning

# Step 5
# train the tuned model and save it
Tuned_SVM.trainSaveModel()

#step 6
# Train and validate results in BERT model
#Please refer BERT_model.py for the same