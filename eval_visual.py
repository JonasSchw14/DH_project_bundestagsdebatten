from main import model_path
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from main import docs, dates
from nltk.corpus import stopwords

if __name__=='__main__':

    # load model from (personal) model path
    model = BERTopic.load(r"C:\Users\jonas\PycharmProjects\DH_topic_modelling\bertopic_test\models\2025-03-19_21_50_10")

    # create topics over times with timestamps
    topics_over_time = model.topics_over_time(docs, dates)

    #remove stopwords after fitting the model (optional)
    german_stop_words = stopwords.words('german')
    vectorizer_model = CountVectorizer(stop_words=german_stop_words, max_df=0.3, min_df=0.1)
    model.update_topics(docs, vectorizer_model=vectorizer_model)

    # visualize topics and save as .html
    fig_topics = model.visualize_topics()
    fig_topics.write_html(os.path.join(model_path + "visualization.html"))

    #visualize topics over time
    fig_topics_over_time = model.visualize_topics_over_time(topics_over_time)
    fig_topics_over_time.write_html(os.path.join(model_path + "topics_over_time.html"))
