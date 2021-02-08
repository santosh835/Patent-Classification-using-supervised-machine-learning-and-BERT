import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def trainSaveModel():
    #reading the clean dataframe 
    df=pd.read_csv("data/final_clean_data.csv")
    df["patentDetails"]=df["description"]
    
    #df["patentDetails"]=df["claims"]
    col = ['classifications', 'patentDetails']
    df = df[col]
    df = df[pd.notnull(df['patentDetails'])]
    
    
    
    ## Training the final model
    #####################################################
    df['class_id'] = df['classifications'].factorize()[0]
    category_id_df = df[['classifications', 'class_id']].drop_duplicates().sort_values('class_id')
    
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.patentDetails).toarray()
    labels = df.class_id
    features.shape
    
    
    model = LinearSVC(class_weight='balanced',C=10)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.25, random_state=0,stratify=df.classifications.values)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.classifications.values, yticklabels=category_id_df.classifications.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    
    from sklearn import metrics
    print(metrics.classification_report(y_test, y_pred, target_names=df['classifications'].unique()))
    
    
    import _pickle as cPickle
    # save the classifier
    with open('model/tuned_SVM_classifier.pkl', 'wb') as fid:
        cPickle.dump(model, fid)    


