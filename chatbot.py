#!/usr/bin/env python3


try:
   mode = int(input('Do you want to talk? 1 for Y | any other number for N:  '))
   if mode == 1:
      pass
   else:
      exit()
except ValueError:
    print("It is not a number, so I dont' understand... quitting")
    exit()

print("Ok! let's start! If you want to quit type  \\q \n\n")

from chatbot_model import weight_matrix
from chatbot_model import questions_tokens
from chatbot_model import dense_tfidf_matrix
from chatbot_model import weight_matrix
from chatbot_model import predict_answer

stop = False
while stop is False:
   question = input('Talk to me\n    --> ')
   if question == "\\q":
      stop = True
   else:
      answer, similarity = predict_answer(question, questions_tokens, dense_tfidf_matrix, weight_matrix, verbose=False)
      print("    --> {}".format(answer))
      print("    (similarity is {})".format(answer))

