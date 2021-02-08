Patent Classification and Analysis
To run the entire classification flow please run start.py

# Step 1
#Read contained xml file
#files are downloaded from  https://bulkdata.uspto.gov/
# Parse the required information from xml like patent title,abstract,
#description,claims

USE USPTOParser.py for the same

#step 2
#clean the datset and save it into csv for model training

run dataCleaning.py for the same

#Step 3 
#Perfrom Exploratory data analysis

Refer EDA_Patent.py for the same

#Step 4 
#Evaluate different models and do hyperparameter tuning

please refer Multiclass_Patent_Classification.py for the same

# Step 5
# train the tuned model and save it
Tuned_SVM.trainSaveModel()


#Step 6
To train and save model using Biderectional encoded representation of the transformers
please refer BERT_model.py file

Model description
BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.
Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.

