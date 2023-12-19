#DO ALL NECESSARY IMPORTS
import ast
import difflib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import CamembertConfig, CamembertModel, CamembertTokenizer, CamembertTokenizer, CamembertForSequenceClassification
from transformers import BertModel, BertTokenizer
import torch
import requests
import io
import pdfplumber


# read in your training data
class Preprocess(): 
    def __init__(self, df):
        self.data = df
        self.data.dropna()
        self.data.drop_duplicates()
        self.sp = spacy.load('fr_core_news_lg')
        self.spacy_stopwords = spacy.lang.fr.stop_words.STOP_WORDS
        self.corpus = df['sentence'].tolist()
        self.french_cognates = list()
        pass
    #method that tokenize, takes out stopwords, and counts token in df
    def tokenize_stop_words_count(self):
        self.data['sentence_sp'] = self.data['sentence'].apply(self.sp)
        self.data['tokens'] = self.data['sentence_sp'].apply(lambda doc: [token.text for token in doc])
        self.data['tokens_no_stop'] = self.data['tokens'].apply(lambda tokens: [token for token in tokens if token.lower() not in self.spacy_stopwords])
        self.data['token_count_no_stop'] = self.data['tokens_no_stop'].apply(len)
        self.data['token_count'] = self.data['tokens'].apply(len)
        #return df
    #method that counts selectted pos
    def count_verbs_nouns_adj(self):
        self.data['nb_verbs'] = self.data['sentence_sp'].apply(lambda x: sum(1 for token in self.sp(x) if token.pos_ == 'VERB'))
        self.data['nb_nouns'] = self.data['sentence_sp'].apply(lambda x: sum(1 for token in self.sp(x) if token.pos_ == 'NOUN'))
        self.data['nb_adj'] = self.data['sentence_sp'].apply(lambda x: sum(1 for token in self.sp(x) if token.pos_ == 'ADJ'))
        self.data['nb_adv'] = self.data['sentence_sp'].apply(lambda x: sum(1 for token in self.sp(x) if token.pos_ == 'ADV'))
    #return df
        
    #methods that compute tfidf score of each sentence
    def tfidf_sentence_unigram(self):
        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=list(self.spacy_stopwords))
        features = tfidf.fit_transform(self.corpus)
        results = pd.DataFrame(features.todense(),columns=tfidf.get_feature_names_out(),)
        word_freq = results.sum().sort_values(ascending=False)
        self.data['words'] = self.data['sentence'].apply(lambda x: x.lower().split())
        self.data['tfidf_score_unigram'] = self.data['words'].apply(lambda words: sum(word_freq.get(word, 0) for word in words))
        #return df

    def tfidf_sentence_bigram(self):
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(self.spacy_stopwords))
        features = tfidf.fit_transform(self.corpus)
        results = pd.DataFrame(features.todense(),columns=tfidf.get_feature_names_out(),)
        word_freq = results.sum().sort_values(ascending=False)
        self.data['words'] = self.data['sentence'].apply(lambda x: x.lower().split())
        self.data['tfidf_score_bigram'] = self.data['words'].apply(lambda words: sum(word_freq.get(word, 0) for word in words))
        #return df
    def find_cognates(self, french_cognates, similarity_threshold=0.90):
        token_list = ast.literal_eval(self.data['tokens'])
        i = 0
        for french_word in french_cognates:
            for words in token_list:
                similarity = difflib.SequenceMatcher(None, french_word, words).ratio()
                if similarity > similarity_threshold:
                    i+=1
        return i
    def cognates_similarities(self):
        self.get_cognates
        self.data['cognate_count'] = self.data['tokens'].apply(lambda x: self.find_cognates(x, self.french_cognates))
    def encoder(self):
        label_encoder = LabelEncoder()
        self.data['encoded_diff'] = label_encoder.fit_transform(self.data['difficulty'])
    def get_cognates(self):
        pdf_url = 'https://docs.steinhardt.nyu.edu/pdfs/metrocenter/xr1/glossaries/ELA/GlossaryCognatesFrenchUpdated5-5-2014.pdf'
        response = requests.get(pdf_url)
        response.raise_for_status()
        with io.BytesIO(response.content) as open_pdf_file:
            with pdfplumber.open(open_pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    for line in text.split('\n'):
                        parts = list(filter(None, line.split(' ')))
                        if len(parts) == 4:
                            cognates.append((parts[1], parts[3]))
        cognates = pd.DataFrame(cognates, columns = ['1', '2'])
        cognates = cognates[cognates.apply(lambda x: x[0][0].lower() == x[1][0].lower(), axis=1)]
        previous_word = 'a'
        for index, row in cognates.iterrows():
            #print(row[0][0], previous_word)
            if row[0][0] == 'a' and previous_word[0] == 'v':
                first_french = index
            else:
                previous_word = row[0]
        self.french_cognates = cognates['1'][first_french-14:].tolist()
    def get_camembert_pooled_embedding(self, model_name = 'camembert-base'):
        
        camembert_model = CamembertModel.from_pretrained(model_name)
        tokenizer = CamembertTokenizer.from_pretrained(model_name)
        #tokenizer = CamembertTokenizer.from_pretrained(model_name, revision="main", sentencepiece_model="/usr/local/lib/python3.10/dist-packages")
        
        def get_bert_embedding(sentence):
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = camembert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()
        def max_pool_embeddings(embeddings):
            return np.max(embeddings, axis=0)

        self.data['camembert_embedding'] = self.data['sentence'].apply(get_bert_embedding)
        self.data['cam_pooled_embedding'] = self.data['camembert_embedding'].apply(max_pool_embeddings)
    
    def data_preprocess(self):
        self.data.tokenize_stop_words_count
        self.data.count_verbs_nouns_adj
        self.data.tfidf_sentence_unigram
        self.data.tfidf_sentence_bigram
        self.data.cognates_similarities