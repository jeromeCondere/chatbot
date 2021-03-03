from pprint import pprint
import nltk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from nltk.tag import pos_tag
from processTrainingSet import normalize_sentence
from datetime import datetime


spacy = en_core_web_sm.load()

input_sentence = 'How do you like my pasta?'

def get_spacy_info(sentence):
  return spacy(sentence)

def print_entity(doc):
  pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])


def from_token_to_tags(tokens):
  tags = nltk.pos_tag(tokens)
  return tags

def get_doc_info(doc):
  return [(X,X.lemma_, X.ent_iob_, X.ent_type_,X.tag_) for X in doc]


doc = spacy('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint(get_doc_info(doc))
displacy.serve(doc, style="ent")


# print(from_token_to_tags(normalize_sentence('when are you')))