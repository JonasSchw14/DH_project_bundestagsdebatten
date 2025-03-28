import pathlib
import os
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
import datetime
import numpy as np
from data import output_folder
from nltk.corpus import stopwords

project_path = r'C:\Users\jonas\OneDrive\Dokumente\Master Data Science\1. Semester\Digital Humanities\project'
#define model path
working_directory = str(pathlib.Path().resolve())
model_path = os.path.join(project_path, "models",datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))

#load documents
doc_path = os.path.join(output_folder, "speeches_linke.npy")
docs_and_dates = np.load(doc_path)

docs = list(docs_and_dates[0])
dates = list(docs_and_dates[1])


if __name__ == "__main__":
    # Create representation model
    representation_model = KeyBERTInspired()

    #embedding model
    embedding_model = "all-mpnet-base-v2" # option: all-MiniLM-L6-v2"

    # cluster model
    # the parameter 'min_cluster_size' determines indirectly how many clusters are build
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf')


    # insert list of german stopwords to remove them
    german_stop_words = stopwords.words('german')

    # vectorizer model with german stop words
    vectorizer_model = CountVectorizer(stop_words=german_stop_words, max_df=0.3, min_df=0.1)

    #ctfidf model
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # create multilanguage bertopic model
    model = BERTopic(ctfidf_model= ctfidf_model,representation_model=representation_model,vectorizer_model=vectorizer_model,embedding_model=embedding_model,hdbscan_model=hdbscan_model,language="multilanguage")

    #fit model to documents
    topics, probs = model.fit_transform(docs)

    #save model in model_path
    model.save(model_path,serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

