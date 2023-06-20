# cleaning-Reddit-WSB-comments-Kaggle-dataset
In this code I am cleaning the Wall street bets comment dataset found on Kaggle here https://www.kaggle.com/datasets/mattpodolak/reddit-wallstreetbets-comments
The aim of this cleaning is to align mis-aligned columns and to prepare the data for machine learning by doing things like, 
 deleting NaN and bad data
 removing special characters
 converting to lower case
 removing extra whitespace
 removing stop words
 lemmatizing words
 tokenizing the text
 dropping data that is not relevant for topic modling, ie comments like "yes", "no"
