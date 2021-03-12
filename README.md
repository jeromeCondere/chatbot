# Chatbot

This project is a simple chatbot using GloVe for word representation as well as an Tfidf vectorizer in order to get an
accurate reprensation of the sentences provided insid the dialog file

## Main idea
Inside the dialogs file there is on each line one question or statement followed by the answer.\
The idea is everytime the user type something, we look for the closest matching questions/statement in the file
and we use the answer associated to that match.\
\
In order to check similarity between sentences we use the cosine measure.\
This is a fairly decent approach however, this methods doesn't extract the semantic of the sentences processed.
There are other methods for that.