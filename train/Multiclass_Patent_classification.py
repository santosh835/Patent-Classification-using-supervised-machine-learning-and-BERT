import pandas as pd

#Loading additional helper functions not shown here but provided in the folder

'''Features'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

'''Classifiers'''
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

'''Display'''
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format


#reading the clean dataframe 
df=pd.read_csv("final_clean_data.csv")

#df["patentDetails"]=df["patent_title"] + df["abstracts"] + df["claims"] + df["description"]
df["patentDetails"]=df["description"]

#df["patentDetails"]=df["claims"]
col = ['classifications', 'patentDetails']
df = df[col]
df = df[pd.notnull(df['patentDetails'])]


#Creating the features (tf-idf weights) for the processed text
texts = df['patentDetails'].astype('str')
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), 
                                   min_df = 5)
X = tfidf_vectorizer.fit_transform(texts) #features
y = df['classifications'].values #target
print (X.shape)
print(y.shape)

#Dimenionality reduction. Only using the 100 best features er category
lsa = TruncatedSVD(n_components=100, 
                   n_iter=10, 
                   random_state=3)

X = lsa.fit_transform(X)
X.shape



#Preliminary model evaluation using default parameters
#Creating a dict of the models
model_dict = {
              'Random Forest': RandomForestClassifier(random_state=3),
              'Gaussian Naive Bayes': GaussianNB(),
              'K Nearest Neighbor': KNeighborsClassifier(),
              'Support vector linear':LinearSVC()}

#Train test split with stratified sampling for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = .25, 
                                                    shuffle = True, 
                                                    stratify = y, 
                                                    random_state = 3)

#Function to get the scores for each model in a df
def model_score_df(model_dict) :
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k,v in model_dict.items():   
        model_name.append(k)
        v.fit(X_train, y_train)
        y_pred = v.predict(X_test)
        ac_score_list.append(accuracy_score(y_test, y_pred))
        p_score_list.append(precision_score(y_test, y_pred, average='macro'))
        r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
        model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
        model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
    return model_comparison_df

model_score_df(model_dict)


#Gridsearch with 5-fold cross validation
print("Performing grid search ... ")
# Parameter Grid
param_grid =  [{'C': [0.1, 1, 10, 100,1000]}]

# Make grid search classifier
clf_grid = GridSearchCV(LinearSVC(), param_grid,cv=5, verbose=10,n_jobs=40)

# Train the classifier
clf_grid.fit(X_train, y_train)

# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

