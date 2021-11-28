from altair.vegalite.v4.schema.core import FontStyle
import streamlit as st
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import datetime
from dateutil import parser
import altair as alt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


# Importing Data
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

    return filtered_sentence, w


st.title("MMA Text Analysis")

d = st.date_input(
    "Choose a Date Range",
    value=(datetime.datetime(2021, 11, 19), datetime.datetime(2021, 11, 28)),
    min_value=datetime.datetime(2021, 11, 19),
    max_value=list(data['Date'].sort_values(ascending=False))[0].date(),
)

subreddits = st.multiselect('Subreddits Chosen', ['MMA', 'ufc'], ['MMA', 'ufc'])

data = data[data['subreddit'].isin(subreddits)]

data['sent'] = data['title'].apply(lambda x: sid.polarity_scores(x)['compound'])

st.markdown("## Negative Sentiment Posts:")
for i in data.sort_values(by='sent', ascending=True)['title'].index[0:5]:
    st.markdown("* " + str(data.sort_values(by='sent', ascending=True)['title'][i]))

st.markdown("## Positive Sentiment Posts:")
for i in data.sort_values(by='sent', ascending=True)['title'].index[-6:-1]:
    st.markdown("* " + str(data.sort_values(by='sent', ascending=True)['title'][i]))

data = data[
    (data["Date"] > parser.parse(str(d[0]))) & (data["Date"] < parser.parse(str(d[1])))
]
filtered_sentence = phrases_and_bigrams(data)[0]
filtered_text = phrases_and_bigrams(data)[1]

st.markdown("## Most Popular Threads:")
for i in data.sort_values(by='score', ascending=False)['title'].index[0:5]:
    st.markdown("* " + str(data['title'][i]))

st.markdown("## Most Common Tokens:")
common_tokens = pd.DataFrame(Counter(filtered_sentence).most_common(5)).rename(
    columns={0: "Token", 1: "Count"}
)
common_tokens.index = common_tokens['Token']

bigrams = pd.DataFrame(
    Counter(list(nltk.bigrams(filtered_sentence))).most_common(5)
).rename(columns={0: "Bigram", 1: "Count"})

st.table(common_tokens['Count'])


st.markdown("## Most Common Bigrams:")
st.table(bigrams)

# c = alt.Chart(common_tokens).mark_bar().encode(x = 'Count', y='Token').configure_axis(
#     labelFontSize=25,
#     titleFontSize=35
# )
# st.altair_chart(c)

# b = alt.Chart(bigrams).mark_bar().encode(x = 'Count', y='Count').configure_axis(
#     labelFontSize=25,
#     titleFontSize=35)

# st.altair_chart(b)

base = alt.Chart(data).encode(
    x=alt.X('x:O', axis=None),
    y=alt.Y('y:O', axis=None)
).configure_view(strokeWidth=0)  # remove border

word_cloud = base.mark_text(baseline='middle').encode(
    text='title',
    color=alt.Color('count:Q', scale=alt.Scale(scheme='goldred')),
    size=alt.Size('count:Q', legend=None)
)

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(filtered_text)

st.markdown('## Word Cloud:')
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
