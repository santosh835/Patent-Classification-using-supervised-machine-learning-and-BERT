import pandas as pd


df=pd.read_csv("data/final_clean_data.csv")


#downsampling majority classes from  1000 to 500 
a1=df.loc[df['classifications'] == 'TEXTILES']
df=df.loc[~df['classifications'].isin(['TEXTILES'])]
df=df.groupby('classifications').apply(lambda x: x.sample(500)).reset_index(drop=True)
df=pd.concat([df, a1], ignore_index=True)
df.groupby(['classifications']).count()
df=df.sample(frac=1)


df.info()

df.classifications.value_counts()


df["patentDetails"]=df["description"]
col = ['classifications', 'patentDetails']
df = df[col]
df = df[pd.notnull(df['patentDetails'])]


def word_count(text):
    return len(str(text).split(' '))


#Avg word count by category
df['word_count'] = df['patentDetails'].apply(word_count)
avg_wc = df.groupby('classifications').mean().reset_index()
avg_wc[['classifications','word_count']]


df['class_id'] = df['classifications'].factorize()[0]
category_id_df = df[['classifications', 'class_id']].drop_duplicates().sort_values('class_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['class_id', 'classifications']].values)
df.head()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('classifications').patentDetails.count().plot.bar(ylim=0)
plt.show()

#convert features into tfidf vectors
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.patentDetails).toarray()
labels = df.class_id
features.shape

#check the most corelated terms in each class
from sklearn.feature_selection import chi2
import numpy as np
N = 2
for classifications, class_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == class_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(classifications))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))



