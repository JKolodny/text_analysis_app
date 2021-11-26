import streamlit as st
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import datetime
from dateutil import parser
nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv("r_mma_post.csv")
data = data[["subreddit", "title", "score", "upvote_ratio", "Date"]]
data = data[data["Date"].notnull()]
data["Date"] = data[data["Date"].notnull()]["Date"].apply(lambda x: parser.parse(x))

# NLP
def phrases_and_bigrams(df) -> list:

    title_text = df["title"]
    title_text = title_text.apply(nltk.tokenize.word_tokenize)

    w = ""
    for comment in title_text:
        for word in comment:
            w += " " + word

    tokenizer = RegexpTokenizer(r"\w+")
    word_tokens = tokenizer.tokenize(w.lower().strip())

    stop_words = set(stopwords.words("english"))
    stop_words.update(
        [
            "ufc",
            "vs",
            "main",
            "fights",
            "fight",
            "thread",
            "discussion",
            "event",
            "amp",
            "spoiler",
            "card",
            "official",
            "night",
            "would",
            "general",
            "fighters",
            "never",
            "think",
            "like"
            "gon",
            "got"
            "done",
            "says",
            "mma",
            "november",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
        ]
    )

    filtered_sentence = [w for w in word_tokens if not w.lower().strip() in stop_words]

    return filtered_sentence


st.title("MMA Text Analysis")

d = st.date_input(
    "Choose a Date Range",
    value=(datetime.datetime(2021, 11, 19), datetime.datetime(2021, 11, 26)),
    min_value=datetime.datetime(2021, 11, 19),
    max_value=datetime.datetime(2021, 11, 26),
)

subreddits = st.multiselect('Subreddits Chosen', ['MMA', 'ufc'])

data = data[data['subreddit'].isin(subreddits)]

data = data[
    (data["Date"] > parser.parse(str(d[0]))) & (data["Date"] < parser.parse(str(d[1])))
]
filtered_sentence = phrases_and_bigrams(data)

st.write('Most Popular Threads:')
st.table(data.sort_values(by='score', ascending=False)['title'].head(5))

st.write("Most Common Tokens:")
common_tokens = pd.DataFrame(Counter(filtered_sentence).most_common(5)).rename(
    columns={0: "Token", 1: "Count"}
)
bigrams = pd.DataFrame(
    Counter(list(nltk.bigrams(filtered_sentence))).most_common(5)
).rename(columns={0: "Bigram", 1: "Count"})
st.table(common_tokens)
st.write("Most Common Bigrams:")
st.table(bigrams)


