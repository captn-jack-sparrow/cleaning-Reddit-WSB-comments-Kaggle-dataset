# cleaning-Reddit-WSB-comments-Kaggle-dataset
In this code I am cleaning the Wall street bets comment dataset found on Kaggle here https://www.kaggle.com/datasets/mattpodolak/reddit-wallstreetbets-comments





import nltk
nltk.download('wordnet')
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\Obama\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
True
import nltk
nltk.download('punkt')
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\Obama\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
True
import nltk
​
nltk.download('stopwords')
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\Obama\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
True
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# In the below code we are importing the data from where it is stored in the state of a csv file.
# We will be chunking the data, this makes it easier on my computer and avoids memory errors
# We will also be processing this data 1.5 million rows at a time, as you may notice by looking at the
# nrows=1500000 in our pd.read_csv() command. 
# This is done to avoid the memory limitations of ram on my laptop. We will preprocess the data 1 million rows at a time
# save those rows to a file, rinse and repeat for the total 34 million rows until the whole dataset is cleaned. 
# from there we will concatenate the data to one data frame and train our model on it.
# , skiprows=range(1, 32000000)
​
chunk_size = 200000
df_chunks = pd.read_csv("C:/Users/pathway/.csv", chunksize=chunk_size, encoding="ISO-8859-1",skiprows=range(1, 29000001), nrows=1500000)
df_fm = pd.concat(df_chunks, ignore_index=True)
# Exploring the dataset, lots of the columns are misaligned, after about every 500k or 1 million rows the columns  
# shift a couple positions to the left or right. This happened because the author who got the data got it from the
# Reddit API and called the data 500k or 1 million rows at a time; he then concatenated the whole dataset into a unclean 
# csv file.
​
# To resolve this issue I will manually go through the dataset and inspect the columns, then take note of the shifts and 
# remedy them.
​
# final note, we are still looking at the data 1.5 million rows at a time.
​
# First, inspecting.
# Here we are changing some of the display settings for Jupyter Notebook so we can get a look at our data.
# Using the .head() method we output the data to be seen.
​
pd.set_option('display.max_colwidth', 1000)
pd.set_option("display.max_rows", 1000000)
pd.set_option("display.max_columns", 55)
df_fm.head(90)
ll = df_fm.iloc[499998:501050]
​
ll.head(60)
intra = df_fm.iloc[999980:1000100]
intra.head(150)
inj = df_fm.iloc[1200000:1200120]
inj.head(350)
las = df_fm[1499960:1500000]
las.head(50)
# Rinse and repeat till the end of the dataset.
# Below are the notes on the dataset I took.
# comment p: So in the original dataset the columns are good until row 2 million, at row 2 million and 1, the dataset is shifted to
# the right by one column at the "created_utc" column. To emphasize, "created_utc" at 2m1 is NaN and the real values are
# one to the right at gildings.
# close comment p.
# 
# comment o: At 5,000,000 the pattern stops, at 5,000,001 it reverts back to the original 0-2m pattern and ends at 6 m 
# close comment 0.
# 
# comment i: At 6m1 the rows become unstable but usable, they are shifted around this pattern continues until 7m.
# close comment i.
# 
# comment j: At 7m1 it time is under gildings, and body is under body until  8m
# close comment j.
# 
# comment bb: From 8m to 11m it is body under body time under created_utc, (in other words, normal).
# close comment bb.
#
# comment k: starting at 11m to 12m body is under author_flair_type and time is under author_premium.
# close comment k.
#
# comment h: starting at 12m1 to 14m body is under body and time is under gildings.
# close comment h.
#
# comment g: staring at 14m to 16m body is under body and time is under created_utc.
# close comment g.
#
# comment tt: starting at 16m to 18m body is inder body and time is under gildings.
# close comment tt.
#
# comment pp: starting at 18m to 20m body is under body and time is under created_utc.
# close comment pp.
#
# comment hh: starting at 20m to 21m the body is under body and time is under gildings.
# close comment hh.
#
# comment qi: starting at 21m to 22m body under body and time under created_utc.
# close comment qi.
#
# comment nn: starting at 22m to 24m body under body and time under gildings.
# close comment nn.
# 
​
# standard pull df_chunks = pd.read_csv("C:/Users/Obama/Downloads/kaggle_datasets/Reddit_wsb_data/wsb_comments_raw.csv", chunksize=chunk_size, encoding="ISO-8859-1", skiprows=range(1, 24000000), nrows=1500001)
# error at ParserError: Error tokenizing data. C error: Expected 40 fields in line 25000002, saw 41
# solution, will skip data from 24,999,990 to 25,000,010.
# 
# 
# 24m-24.999,990 has body under author_flair_type and time under author_premium
#
# 25.000,010 to 26m has body under author_flair_text_color and time is under author_patreon_flair (authorname under associated_award)
# 
# 26m to 28m body under author_premium and time under awarders (author still under associated_awards)
# 
# 
# ##############(instance of misalignment) 26.5m to 28m   body u awarders time u body (author u author)##########################
#
#
# ############# more errors ParserError: Error tokenizing data. C error: Expected 40 fields in line 29000002, saw 52 #####
# 
# ###### will do skiprows=range(1, 28000000), nrows=1000001)
#
# 
# regardless of the data mess above. at 28m1 to 29m the data follows body under body time under created_utc and author under author.
#
#
# at 29m10 to 30m body under author time under author_flair_css_class AUTHOR IS LOST
#
# at 30m to end ALL DATA IS LOST
#
#
#
#
# 
# 
# 0-2million it is correct. 2m1-5m that the pattern listed in comment p holds up. 5,000,001 it goes to the original pattern (comment o documents this).
# so far to 6 m the original pattern is good. At 6 million and 1 the pattern follows comment i until 7m. At 7m1 it follows the
# pattern described in comment j until 8m. At 8m to 11m it follows comment bb. Starting at 11m to 12m it follows comment k.
# At 12m1 the pattern follows comment h to 14m. starting at 14m to 16m comment g. starting at 16m to 18 comment tt. starting
# at 18m to 20m comment pp. starting at comment 20m to 21m comment hh. starting at 21m to 22m comment qi. from 22m to 24m? comment
# nn.
#
# Once I went through the dataset I saw two things; the data is usable and the ways the data is misaligned. With this known
# I again opened the datasets 1.5 million rows at time, and then drop the unnecessary columns,
# and finally I would manually rename the columns with their proper names. 
​
#(below)
chunk_size = 200000
df_chunks = pd.read_csv("C:/Users/pathway/.csv", chunksize=chunk_size, encoding="ISO-8859-1",skiprows=range(1, 29000001), nrows=1500000)
df_fm = pd.concat(df_chunks, ignore_index=True)
​
columns_to_drop = ["author_cakeday", "edited", "media_metadata", "comment_type", "author_flair_background_color",
                   "author_flair_css_class", "author_flair_richtext", "author_flair_template_id", "author_flair_text", 
                  "author_flair_text_color", "author_flair_type", "author_patreon_flair", "author_premium", "awarders",
                   "collapsed_because_crowd_control", "gildings", "id", "is_submitter", "link_id", "locked", "no_follow",
                   "parent_id", "permalink", "score", "send_replies", "stickied", "subreddit_id", "treatment_tags", "distinguished",
                   "editable", "retrieved_on", "top_awarded_type", "subreddit", "associated_award", "all_awardings"]
​
uhoh = df_fm.drop(columns=columns_to_drop)
​
new_names = {"gildings":"created_utc", "author_flair":"body"}
​
realinged_df = uhoh.rename(columns=new_names)
​
#
# FYI on each different loaded 1.5 million rows, the step above would be different!!!!!
#
# I would then save the cleaner df to a new file for further cleaning, like so
folder_path = "C:/Users/pathway/"
file_name = "WSB7-7.5M.csv"
​
file_path = folder_path + file_name
​
realinged_df.to_csv(file_path, index=False)
# Now that the dataset's data is under the right columns we can clean it for machine learning
# Also, doing this manually took a long time :( , so I tried making it more automated.
# In the below code we are:
​
# deleting NaN and bad data
​
# removing special characters
​
# converting to lower case
​
# removing extra whitespace
​
# removing stop words
​
# lemmatizing words
​
# tokenizing the text
​
# dropping data that is not relevant for topic modling, ie comments like "yes", "no"
files = ["WSB999,999-1.5M", "WSB1.5-2M", "WSB2-3M", "WSB3-4.5M", "WSB4.5-5M", "WSB5-6M", "WSB6-7M", "WSB7-7.5M", "WSB7.5-8M", "WSB8-9M", "WSB9-10M", "WSB10-10.5M", "WSB10.5-11M", "WSB11-12M", "WSB12-13.5M", "WSB13.5-14M", "WSB14-15M", "WSB15-16M", "WSB16-16.5M", "WSB16.5-18M", "WSB18-19M", "WSB19-19.5M", "WB19.5-20M", "WSB20-21M", "WSB21-22M", "WSB22-22.5M", "WSB22.5-24M", "WSB24-24.999,990M", "WSB25.000.010-26M", "WSB26-27.5M", "WSB27.5-28M", "WSB28-29M"]
for file in files:
    df_nks = pd.read_csv(f"C:/Users/pathway/{file}.csv")
    df = pd.DataFrame(df_nks)
​
    def remove_special_characters(text):
        if isinstance(text, str):  # Check if input is a string
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            return cleaned_text
        else:
            return ''
​
    def convert_to_lowercase(text):
        return text.lower()
​
    def remove_extra_whitespace(text):
        cleaned_text = re.sub(r'\s+', ' ', text)
        return cleaned_text
​
    def remove_stopwords(text):
        if isinstance(text, str) and not pd.isnull(text):
            stop_words = set(stopwords.words("english"))
            tokens = nltk.word_tokenize(text)
            filtered_text = ' '.join([word for word in tokens if word.lower() not in stop_words])
            return filtered_text
        else:
            return np.nan
​
    def lemmatize_words(text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
        return lemmatized_text
​
    df = df.drop(df[(df['body'].isna())].index)
​
    df['body'] = df['body'].apply(remove_special_characters)
​
    df['body'] = df['body'].apply(convert_to_lowercase)
​
    df['body'] = df['body'].apply(remove_extra_whitespace)
​
    df['body'] = df['body'].apply(remove_stopwords)
​
    df['body'] = df['body'].apply(lemmatize_words)
​
    words_delete = ["removed", "deleted", " ", "  ", "   "]
    author_fullname = pd.NA
​
​
    df = df.drop(df[(df['body'].isin(words_delete)) | (df['author_fullname'].isna())].index)
    df = df.drop(df[df["author_fullname"].isin(words_delete)].index)
    df = df.drop(df[(df['created_utc'].isin(words_delete)) | (df['created_utc'].isna())].index)
    df = df.drop(df[(df['author'].isin(words_delete)) | (df['author'].isna())].index)
    df = df.drop(df[(df['total_awards_received'].isin(words_delete)) | (df['total_awards_received'].isna())].index)
​
    from nltk.tokenize import word_tokenize
​
    def tokenize_text(text):
        tokens = word_tokenize(text)
        return tokens
​
    df['body'] = df['body'].apply(lambda x: tokenize_text(x))
​
    useless = ["", "yes", "ban", "way", "lol", "f", "nice", "fuck", "link", "ok", "know",
               "rip", "thanks", "gay", "nope", "hope", "guh", "retarded", "stonks, go", "retard", "position, ban",
               "eat, dongus, fuckin, nerd, bot, action, performed, automatically, please, contact, moderator, subreddit, message, compose, r, wallstreetbets, question, concern", 
              "buy", "fact", "right", 
               "post, flaired, dd, dd, list, find, fresh, wsb, dd, http, n, reddit, com, r, wallstreetbets, search, sort, new, amp, q, flair, 3add, amp, restrict, sr, amp, da, misuse, dd, flair, shitposts, short, vague, guess, unexplained, news, link, etc, please, change, flair, dd, mod, notified, thread, sure, flair, use, check, guide, post, flair, http, www, reddit, com, r, wallstreetbets, wiki, linkflair, bot, action, performed, automatically, please, contact, moderator, subreddit, message, compose, r, wallstreetbets, question, concern"
              "good", "15, monday", "future", "yep", "next, week", "always", "say, take, see, hold, gain, nobody, left", "nah", "bruh", 
              "oof", "real", "please, resubmit, shorter, title, bot, action, performed, automatically, please, contact, moderator, subreddit, message, compose, r, wallstreetbets, question, concern", 
              "true", "thank", "rh", "bb", "say", "earnings", "go", "like", "broken, spoke, flair, plz, mod", "priced", "tldr", "yessir", 
              "maybe", "sir, bread, line, bot, action, performed, automatically, please, contact, moderator, subreddit, message, compose, r, wallstreetbets, question, concern",
              "username, check", "exactly", "lmao", "son, bitch", "b", "lolol, future, barely, green"]
​
    df = df.drop(df[(df['body'].isin(useless))].index)
​
    final_df = df.drop("author_fullname", axis=1)
    
    folder_path = "C:/Users/pathway/"
    file_name = file
    csv_postfix = ".csv"
    file_path = folder_path + file_name + csv_postfix
    final_df.to_csv(file_path, index=False)
    
​
