#!/usr/bin/env python3

import pandas as pd 
import numpy as np
from glove import Glove
from nltk.tokenize import word_tokenize
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()


def parse_training_file(file):
  with open(file) as tsv:
    parsed = [line.replace('\n', '').split('\t') for line in tsv]
  return [x[0] for x in parsed], [x[1] for x in parsed]


def read_glove_file(file):
  glove_vectors = dict()
  with open(file) as glove:
    for line in glove:
      values = line.split()
      words = values[0]
      vectors = np.array([float(i) for i in values[1:]])
      glove_vectors[words] = vectors
  return glove_vectors


def tokenize(questions):
  return [word_tokenize(question) for question in questions]

  
def get_tfidf_dense_matrix(questions):
  vectors = vectorizer.fit_transform(questions)
  feature_names = vectorizer.get_feature_names()
  dense = vectors.todense()
  denselist = dense.tolist()
  df = pd.DataFrame(denselist, columns=feature_names)
  return df


def get_weight_questions_token(questions_tokens, dense_tfidf_matrix):
  weight_matrix = []
  for index, tokens in enumerate(questions_tokens):
    weight_vector = []
    for token in tokens:
      if token in dense_tfidf_matrix.columns:
        weight_vector.append(dense_tfidf_matrix[token][index])
      else:
        weight_vector.append(1.0)
    weight_matrix.append(weight_vector)
  return weight_matrix

# using the information we have from the dense_tfidf_matrix we compute the weight vector
def get_weights_new_sentences(sentence_tokens, dense_tfidf_matrix):
  weight_vector = []
  for token in sentence_tokens:
    if token in dense_tfidf_matrix.columns:
      not_null_vector_token = dense_tfidf_matrix[token][dense_tfidf_matrix[token] > 0]
      weight_vector.append(np.average(not_null_vector_token))
    elif token in glove_vectors:
      # we don't wan those weight vector to be longer than the on we use to
      # make the np.average
      weight_vector.append(1.0)
  return weight_vector

# check the distance between two sentences using cosine similarity and glove for word representation
def cosine_distance_sentence(sent1, sent2, weights1, weights2):
  vector_1 = np.average([glove_vectors[word] for word in sent1 if word in glove_vectors], axis=0, weights=weights1)
  vector_2 = np.average([glove_vectors[word] for word in sent2 if word in glove_vectors], axis=0, weights=weights2)
  cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
  return 1 - cosine


# check the questions closest to the sentence and return the right answer
def predict_answer(sentence, questions_tokens, dense_tfidf_matrix, weight_matrix, in_tolerance=0.95, verbose=True):
  sentence_tokens = word_tokenize(sentence)
  sentence_weight = get_weights_new_sentences(sentence_tokens, dense_tfidf_matrix)
  question_index = -1
  similarity = 0
  
  for index, tokens in enumerate(questions_tokens):
    weight_vector_tokens = weight_matrix[index]
    cosine_distance = cosine_distance_sentence(tokens, sentence_tokens, weight_vector_tokens, sentence_weight)
    if cosine_distance > similarity:
      similarity = cosine_distance
      question_index = index
  
  if verbose is True:
    print('The question closest to the sentence provided is "{}"'.format(questions[question_index]))
  
  if similarity >= in_tolerance:
    return answers[question_index], similarity
  else:
    return "I don't understand, can you reformulate", similarity



questions, answers = parse_training_file('dialogs')
questions_tokens = tokenize(questions)
glove_vectors = read_glove_file('glove.6B.50d.txt')

dense_tfidf_matrix = get_tfidf_dense_matrix(questions)
weight_matrix = get_weight_questions_token(questions_tokens, dense_tfidf_matrix)
