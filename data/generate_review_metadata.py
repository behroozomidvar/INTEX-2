# This library generates the signature for reviews, i.e., topics, tags, and sentiments.

# This library provides access to the data for exploration (e.g., Amazon Reviews).
from concurrent import futures
from concurrent.futures.thread import ThreadPoolExecutor
from numpy.core.records import array
from data import db_interface
from data.db_interface import DBInterface
import re
import math
import random
import gensim
import numpy as np
from gensim import corpora
import pickle
import nltk
from nltk.corpus import wordnet as wn, stopwords
from scipy import spatial
import spacy
from spacy.lang.en import English
from textblob import TextBlob
import pandas as pd
from tqdm import tqdm
nltk.download('wordnet')
nltk.download('stopwords')


class Topic():

    # The class "topic" is instantiated using the number of topics (by default, it is
    # ... set to 10) and the data under exploration.
    def __init__(self, element_id_text_tuples, db_interface: DBInterface, num_topics=10):
        self.element_id_text_tuples = element_id_text_tuples
        self.num_topics = num_topics
        self.db_interface = db_interface
        # The language for the topic modeling (i.e., English)
        spacy.load('en_core_web_lg')

    # The function returns True if all generated topics are successfully inserted into
    # ... the database for all the reviews in the exploration data.
    def generate_topics(self):

        # Obtain corpus and dictionary over the collection of reviews in the exploration data
        corpus, dictionary = self.generate_corpus()

        # Build the LDA topic model using the corpus and the dictionary
        ldamodel = self.define_topic_model(corpus, dictionary)

        # Obtain top-50 words for each calculated LDA topic
        topic_words = ldamodel.print_topics(num_words=50)

        # Truncate the table of topic definitions in the database, in case a previous version
        # ... was already inserted.
        topic_definition_truncate_query = self.db_interface.get_query(
            "topic_definition_truncate")
        topic_definition_truncate = self.db_interface.manipulation_query(
            topic_definition_truncate_query)

        # Insert the top-50 words into the topic definitions in the database.
        for topic_iterate in topic_words:
            topic_definition_query = self.db_interface.get_query(
                "topic_definition_insert", [topic_iterate[0], topic_iterate[1].replace("'", "''")])
            topic_definition = self.db_interface.manipulation_query(
                topic_definition_query)

        element_id_topics_tuples = []
        # Update topics for elements into the database
        print("Updating element topics")
        for index, element_id_text_tuple in tqdm(enumerate(self.element_id_text_tuples)):
            element_topics = list(
                map(lambda x: x[1], ldamodel[corpus[index]]))
            self.db_interface.manipulation_query(
                f"UPDATE elements SET topics=ARRAY{element_topics} WHERE id={element_id_text_tuple[0]}")
            element_id_topics_tuples.append(
                (element_id_text_tuple[0], element_topics))

        # self.generate_topic_relevance(element_id_topics_tuples)

    # Build a textual corpus and a textual dictionary based on a collection
    # ... of textual data.
    # Note: The function should be completed in the following way: in case
    # ... the corpus and dictionary are already built, they don't need to be
    # ... built again, and they just need to be loaded from a saved file.

    def generate_corpus(self):

        # Obtain collection of textual data
        text_data_collection = self.generate_text_data_collection()

        # Build dictionary over the textual data collection
        dictionary = corpora.Dictionary(text_data_collection)

        # Save the dictionary
        dictionary.save('dictionary.gensim')

        # Build corpus over the textual data collection
        corpus = [dictionary.doc2bow(text) for text in text_data_collection]

        # Save the corpus
        pickle.dump(corpus, open('corpus.pkl', 'wb'))

        return corpus, dictionary

    # Generate a "text data collection" by collecting the textual content of
    # ... all reviews in the exploration data.
    def generate_text_data_collection(self):
        text_data_collection = []
        print("Generating corpus")
        for element_id_text_tuple in tqdm(self.element_id_text_tuples):
            # Obtain tokens in the review
            tokens = self.tokenize(element_id_text_tuple[1])
            tokens = [token for token in tokens if len(token) > 4]
            en_stop = set(nltk.corpus.stopwords.words('english'))
            tokens = [token for token in tokens if token not in en_stop]

            # Lemmatize tokens
            tokens = [self.get_lemma(token) for token in tokens]
            text_data_collection.append(tokens)

        return text_data_collection

    # Build LDA topic model
    # Note: The function should be completed in the following way: in case
    # ... the LDA model is already built, it doesn't need to be rebuilt, but
    # ... it just needs to be loaded from a save file.
    def define_topic_model(self, corpus, dictionary):
        # Define model
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=self.num_topics, id2word=dictionary, passes=15, minimum_probability=0.0)
        # Save model
        ldamodel.save('model5.gensim')

        return ldamodel

    # Obtain tokens in the text by removing URLs, spaces, emails, etc.
    def tokenize(self, text):
        parser = English()
        lda_tokens = []
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens

    # Obtain the lemmatezed version of a given word
    def get_lemma(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        return lemma


class Tag():

    # The class "tag" is initialized only using the exploration data.
    def __init__(self, element_id_text_tuples, db_interface: DBInterface):
        self.element_id_text_tuples = element_id_text_tuples
        self.db_interface = db_interface

    # Generate tags for all the reviews in the exploration data
    def generate_tags(self):
        print("Generating tags")
        for element_id_text_tuple in tqdm(self.element_id_text_tuples):
            text = element_id_text_tuple[1]
            # Obtain tags for a given text
            blob = TextBlob(text)
            noun_phrases = list(
                map(lambda x: x.replace("'", ""), blob.noun_phrases))
            self.db_interface.manipulation_query(
                f"UPDATE elements SET tags=ARRAY{noun_phrases}::text[] WHERE id={element_id_text_tuple[0]}")

    # Generate tags using noun phrases
    def get_noun_phrases(text):
        blob = TextBlob(text)
        noun_phrases = blob.noun_phrases
        noun_phrases_separated = ""
        for noun_phrase in noun_phrases:
            noun_phrases_separated += noun_phrase + ";"
        noun_phrases_separated = noun_phrases_separated.replace("'", " ")
        return noun_phrases_separated


class Sentiment():

    # The class "sentiment" is initialized only using the exploration data.
    def __init__(self, element_id_text_tuples, db_interface: DBInterface):
        self.element_id_text_tuples = element_id_text_tuples
        self.db_interface = db_interface

    def generate_sentiments(self):
        print("Generating sentiments")
        for element_id_text_tuple in tqdm(self.element_id_text_tuples):
            # Obtain sentiments for a given review
            element_sentiments = self.get_element_sentiments(
                element_id_text_tuple[1])

            self.db_interface.manipulation_query(
                f"UPDATE elements SET sentiments=ARRAY{element_sentiments} WHERE id={element_id_text_tuple[0]}")

        return True

    # Generate sentiments for a given text
    def get_element_sentiments(self, text):

        blob = TextBlob(text)

        # We bucketize sentiment values using the following 10 buckets: [-inf,-0.8) [-0.8,-0.6)
        # [-0.6, -0.4) [-0.4, -0.2) [-0.2, 0) [0, 0.2) [0.2, 0.4) [0.4, 0.6) [0.6, 0.8) [0.8, +inf]
        sentiment_bucket_vector = [0] * 10

        for sentence in blob.sentences:
            # Compute sentiment value for a given sentence
            sentiment = sentence.sentiment.polarity
            bucket = int(sentiment // 0.2) + 5

        #     # Increase the count of the updated bucket
            sentiment_bucket_vector[bucket] += 1

        return sentiment_bucket_vector


def index_thread_function(elements, threadNumber, thread_boundaries, text_vectors, summary_vectors, tag_vectors, db_dataset ):
    with DBInterface(db_dataset) as db_interface:
        start_index = 0 if threadNumber == 0 else thread_boundaries[threadNumber-1]
        end_index = thread_boundaries[threadNumber]
        print(f"started n{threadNumber} {start_index} {end_index}")
        for index1, element1 in tqdm(elements.iloc[start_index:end_index].iterrows(), total=(end_index-start_index)):
            for index2, element2 in elements.loc[index1+1:(len(elements)-1)].iterrows():
                if element1.type == 'item' or element2.type == 'item' or element1.item_id == element2.item_id:
                    text_similarity = Similarity.cosine_word_embedding(
                        text_vectors[index1], text_vectors[index2])
                    if math.isnan(text_similarity) == True:
                        text_similarity = 0

                    summary_similarity = Similarity.cosine_word_embedding(
                        summary_vectors[index1], summary_vectors[index2])
                    if math.isnan(summary_similarity) == True:
                        summary_similarity = 0

                    topic_similarity = 1.0 - \
                        spatial.distance.cosine(
                            element1.topics, element2.topics)
                    if math.isnan(topic_similarity) == True:
                        topic_similarity = 0

                    sentiment_similarity = 1.0 - \
                        spatial.distance.cosine(
                            element1.sentiments, element2.sentiments)
                    if math.isnan(sentiment_similarity) == True:
                        sentiment_similarity = 0

                    tag_similarity = Similarity.cosine_word_embedding(
                        tag_vectors[index1], tag_vectors[index2])
                    if math.isnan(tag_similarity) == True:
                        tag_similarity = 0

                    attribute_sim = len(element1.attribute_set & element2.attribute_set) / len(
                        element1.attribute_set | element2.attribute_set)

                    db_interface.manipulation_query(f"""INSERT INTO public.similarities(
                        sim, sentiment_sim, tag_sim, topic_sim, attribute_sim, summary_sim, element1_id, element2_id)
                        VALUES ({text_similarity}, {sentiment_similarity}, {tag_similarity}, {topic_similarity}, {attribute_sim}, {summary_similarity}, {element1.id}, {element2.id});""")

class Similarity:
    def __init__(self, language_model_address, db_interface: DBInterface, process_count=4):
        self.language_model = self.load_language_model(language_model_address)
        self.db_interface = db_interface
        self.elements = pd.read_sql(f"""select e.*  
                from elements as e;""", db_interface.conn)
        self.elements["attribute_set"] = self.elements.attributes.apply(
            self.get_attributes_set)
        self.process_count = process_count

    def get_attributes_set(self, attributes):
        attributes_keyvalues = []
        for key, value in attributes.items():
            if type(value) == list:
                attributes_keyvalues += list(
                    map(lambda x: f"{key}:{x}", value))
            else:
                attributes_keyvalues.append(f"{key}:{value}")
        return set(attributes_keyvalues)

    def load_language_model(self, language_model_address):
        # The variable "language_model_content" will contain the whole content of
        # ... the language model file.
        with open(language_model_address, encoding="utf8") as language_model_file:
            language_model_content = language_model_file.readlines()

        # Build language model
        language_model = {}
        for line in language_model_content:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            language_model[word] = embedding

        return language_model

    # This function returns numerical embeddings for the textual content of reviews and their summaries.
    def vectorize(self, texts):
        vectors = []
        count_failure = 0
        print("Vectorizing texts")
        for text in tqdm(texts):
            # try:
            embeddings = [self.language_model.get(
                word, " ") for word in self.preprocess(text)]
            embeddings = [e for e in embeddings if type(e) == np.ndarray]
            vectors.append(np.mean(embeddings, axis=0))
            # except:
            #     count_failure += 1
            #     vectors[text] = self.vectorize("fail")
        return vectors

    # This function cleans the input text by removing unnecessary content.
    def preprocess(self, text):
        letters_only_text = re.sub("[^a-zA-Z]", " ", text)
        words = letters_only_text.lower().split()
        stopword_set = set(stopwords.words("english"))
        cleaned_words = list(set([w for w in words if w not in stopword_set]))
        return cleaned_words

    # Given two vectors of words, this functions computes their similarity score.
    @staticmethod
    def cosine_word_embedding(vector1, vector2):
        cosine = spatial.distance.cosine(vector1, vector2)
        return round((1.0-cosine)*100, 2)

    # This function computes the similarity between pairs of reviews or pairs of summaries.
    # The input parameter "content_type" defines whether the function should operate on
    # ... reviews or their their summaries.
    def generate_summary_review_relevance(self):
        texts = self.elements["text"].to_list()
        text_vectors = self.vectorize(texts)
        summaries = self.elements["summary"].to_list()
        summary_vectors = self.vectorize(summaries)
        tags = self.elements["tags"].to_list()
        tags = list(map(";".join, tags))
        tag_vectors = self.vectorize(tags)
        print("Computing similarities")
        thread_boundaries = self.get_thread_boundaries(
            len(self.elements), self.process_count)
        futures = []
        with ThreadPoolExecutor(max_workers=self.process_count) as executor:
            for i in range(self.process_count):
                futures.append(executor.submit(index_thread_function,
                         elements=self.elements, text_vectors=text_vectors, summary_vectors=summary_vectors, 
                         tag_vectors=tag_vectors, threadNumber=i, thread_boundaries=thread_boundaries, db_dataset=self.db_interface.dataset))

        for future in futures:
            print(future.result())
        # for index1, element1 in tqdm(self.elements.iterrows(), total=self.elements.shape[0]):
        #     for index2, element2 in self.elements.loc[index1+1:(len(self.elements)-1)].iterrows():
        #         if element1.type == 'item' or element2.type == 'item' or element1.item_id == element2.item_id:
        #             text_similarity = self.cosine_word_embedding(
        #                 text_vectors[index1], text_vectors[index2])
        #             if math.isnan(text_similarity) == True:
        #                 text_similarity = 0

        #             summary_similarity = self.cosine_word_embedding(
        #                 summary_vectors[index1], summary_vectors[index2])
        #             if math.isnan(summary_similarity) == True:
        #                 summary_similarity = 0

        #             topic_similarity = 1.0 - \
        #                 spatial.distance.cosine(
        #                     element1.topics, element2.topics)
        #             if math.isnan(topic_similarity) == True:
        #                 topic_similarity = 0

        #             sentiment_similarity = 1.0 - \
        #                 spatial.distance.cosine(
        #                     element1.sentiments, element2.sentiments)
        #             if math.isnan(sentiment_similarity) == True:
        #                 sentiment_similarity = 0

        #             tag_similarity = self.cosine_word_embedding(
        #                 tag_vectors[index1], tag_vectors[index2])
        #             if math.isnan(tag_similarity) == True:
        #                 tag_similarity = 0

        #             attribute_sim = len(element1.attribute_set & element2.attribute_set) / len(
        #                 element1.attribute_set | element2.attribute_set)

        #             self.db_interface.manipulation_query(f"""INSERT INTO public.similarities(
        #                 sim, sentiment_sim, tag_sim, topic_sim, attribute_sim, summary_sim, element1_id, element2_id)
        #                 VALUES ({text_similarity}, {sentiment_similarity}, {tag_similarity}, {topic_similarity}, {attribute_sim}, {summary_similarity}, {element1.id}, {element2.id});""")

    def get_thread_boundaries(self, number_of_elements, number_of_threads):
        boundaries = []
        iterations_per_thread = number_of_elements * \
            (number_of_elements - 1) / 2 / number_of_threads
        set_counter = 0
        iterations_counter = 0
        while len(boundaries) < number_of_threads - 1:
            while iterations_counter < iterations_per_thread:
                set_counter += 1
                iterations_counter = iterations_counter + number_of_elements - 1 - set_counter
            boundaries.append(set_counter)
            iterations_counter = 0
        boundaries.append(number_of_elements)
        return boundaries

    