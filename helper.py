import numpy as np
import pandas as pd
import os
import re
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

analyzer = SentimentIntensityAnalyzer()
extract = URLExtract()


def fetch_stats(selected_user, df):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    # ---------------- BASIC ----------------
    num_messages = df.shape[0]

    words = []
    for message in df["message"]:
        words.extend(message.split())

    # ---------------- MEDIA (UPDATED 🔥) ----------------
    total_media = df["is_media"].sum()
    actual_media = df["has_actual_file"].sum()
    omitted_media = df["is_omitted"].sum()

    # media type counts (only real files)
    media_df = df[df["has_actual_file"]]

    images = media_df[media_df["media_type"] == "image"].shape[0]
    videos = media_df[media_df["media_type"] == "video"].shape[0]
    audio = media_df[media_df["media_type"] == "audio"].shape[0]
    voice = media_df[media_df["media_type"] == "voice"].shape[0]

    # ---------------- LINKS ----------------
    links = []
    for message in df["message"]:
        links.extend(extract.find_urls(message))

    return (
        num_messages,
        len(words),
        total_media,
        actual_media,
        omitted_media,
        images,
        videos,
        audio,
        voice,
        len(links),
    )


# Most Busy User
def most_busy_users(df):
    x = df["user"].value_counts().head()
    df = (
        round((df["user"].value_counts() / df.shape[0]) * 100, 2)
        .reset_index()
        .rename(columns={"index": "name", "user": "percent"})
    )
    return x, df


# Word cloud
def create_wordcloud(selected_user, df):

    # Load stopwords
    with open("stop_hinglish.txt", "r") as f:
        stop_words = set(f.read().split())

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["user"] != "group_notification"]
    temp = temp[temp["message"] != "<Media omitted>\n"]

    def clean_message(message):
        # remove emojis
        message = emoji.replace_emoji(message, replace="")

        # lowercase
        message = message.lower()

        # remove punctuation & numbers
        message = re.sub(r"[^a-zA-Z\s]", "", message)

        # ✅ remove stopwords
        words = [
            word for word in message.split() if word not in stop_words and len(word) > 1
        ]

        return " ".join(words)

    # ⚠️ avoid modifying original dataframe
    cleaned_text = temp["message"].apply(clean_message)

    # ✅ generate wordcloud
    wc = WordCloud(
        width=1200,
        height=500,
        min_font_size=12,
        background_color="white",
        collocations=False,  # 🔥 prevents "bhai bhai" duplication
    )

    df_wc = wc.generate(cleaned_text.str.cat(sep=" "))
    return df_wc


# Most Common words
def most_common_words(selected_user, df):

    with open("stop_hinglish.txt", "r") as f:
        stop_words = set(f.read().split())

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["user"] != "group_notification"]
    temp = temp[temp["message"] != "<Media omitted>\n"]

    words = []

    for message in temp["message"]:

        # Remove emojis
        message = emoji.replace_emoji(message, replace="")

        # Lowercase
        message = message.lower()

        # Remove punctuation & numbers
        message = re.sub(r"[^a-zA-Z\s]", "", message)

        for word in message.split():
            if word not in stop_words and len(word) > 1:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).items(), columns=["word", "count"])

    most_common_df = most_common_df.sort_values(by="count", ascending=False).head(20)

    return most_common_df


# Emoji Analysis
def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    emojis = []
    for message in df["message"]:
        emoji_list = emoji.emoji_list(message)
        emojis.extend([e["emoji"] for e in emoji_list])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


# User Montly and daily timeline
def monthly_timeline(selected_user, df):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = (
        df.groupby(["year", "month_num", "month"]).count()["message"].reset_index()
    )

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline["month"][i] + "-" + str(timeline["year"][i]))

    timeline["time"] = time

    return timeline


def daily_timeline(selected_user, df):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    daily_timeline = df.groupby("only_date").count()["message"].reset_index()

    return daily_timeline


# finding in which week days and months have higher number of chats.
def week_activity_map(selected_user, df):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["day_name"].value_counts()


def month_activity_map(selected_user, df):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["month"].value_counts()


# user Activity Heatmap
def activity_heatmap(selected_user, df):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    user_heatmap = df.pivot_table(
        index="day_name", columns="period", values="message", aggfunc="count"
    ).fillna(0)

    return user_heatmap


# Image
def get_images(selected_user, df, media_folder, limit=20):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    images = df[(df["media_type"] == "image") & (df["has_actual_file"])]

    paths = []
    for file in images["media_file"].head(limit):
        path = os.path.join(media_folder, file)
        if os.path.exists(path):
            paths.append(path)

    return paths


# Video Data
def get_videos(selected_user, df, media_folder, limit=6):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    videos = df[(df["media_type"] == "video") & (df["has_actual_file"])]

    paths = []
    for file in videos["media_file"].head(limit):
        path = os.path.join(media_folder, file)
        if os.path.exists(path):
            paths.append(path)

    return paths


# Voice Data
def get_voice_files(selected_user, df, media_folder, limit=5):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    voices = df[(df["media_type"] == "voice") & (df["has_actual_file"])]

    paths = []
    for file in voices["media_file"].tail(limit):
        path = os.path.join(media_folder, file)
        if os.path.exists(path):
            paths.append(path)

    return paths


# AI Insights
def get_ai_insights(df):

    top_user = df["user"].value_counts().idxmax()
    busiest_day = df["day_name"].value_counts().idxmax()

    media_users = df[df["has_actual_file"]]["user"].value_counts()
    media_user = media_users.idxmax() if not media_users.empty else "N/A"

    return {"top_user": top_user, "busiest_day": busiest_day, "media_user": media_user}


# Sentiment Anlaysis
def add_sentiment(df):

    def get_label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_score"] = df["message"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    df["sentiment"] = df["sentiment_score"].apply(get_label)

    return df


# Similar message
def get_embeddings(model, df):
    return model.encode(df["message"].tolist())


def find_similar(model, query, df, embeddings, top_n=10):
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]

    top_idx = np.argsort(scores)[-top_n:][::-1]
    return df.iloc[top_idx]["message"].tolist()


# Toxicity Detection
bad_words = [
    # Original words
    "bc",
    "mc",
    "madarchod",
    "bhosdike",
    "chutiya",
    # Common Hindi/Hinglish abuses (A-Z coverage)
    "bhosad",
    "bhosdi",
    "bsdk",
    "bhen",
    "behen",
    "behenchod",
    "bkl",
    "bhadwe",
    "bhadwa",
    "chod",
    "chodu",
    "chut",
    "chutad",
    "chutiye",
    "chinal",
    "chinaal",
    "dalle",
    "dalal",
    "dharkan",
    "gandmare",
    "gandu",
    "gaandu",
    "gand",
    "gaand",
    "harami",
    "haramkhor",
    "hijra",
    "kutta",
    "kutte",
    "kamina",
    "kamine",
    "kamini",
    "kuttiya",
    "lodu",
    "lode",
    "laude",
    "lund",
    "lavde",
    "lawde",
    "mkc",
    "maaki",
    "maa ki",
    "maadar",
    "mother",
    "muth",
    "muthal",
    "randi",
    "rnd",
    "rand",
    "raand",
    "randiya" "sala",
    "saala",
    "suar",
    "suwar",
    "teri maa",
    "tere baap",
    "tatto",
    "tatton",
    "ullu",
    "ullu ke pathe",
    # Variations with spaces/symbols
    "m c",
    "b c",
    "m_c",
    "b_c",
    "mc_bc",
    "bc_mc",
    "bhen chod",
    "ma dar chod",
    "lun d",
    "ga and",
    "chu tiya",
    "ran di",
    "har ami",
    "gan du",
    # Leetspeak/number substitutions
    "b3hen",
    "ch0d",
    "g4nd",
    "l0de",
    "m@dar",
    "r@ndi",
    # Asterisk censored versions
    "b*c",
    "m*c",
    "ch*t",
    "l*nd",
    "g*nd",
    "r*ndi",
    "b***d",
    "bh***d",
    "madarch**",
    "behen***",
    # Other common toxic terms
    "bastard",
    "bitch",
    "damn",
    "hell",
    "shit",
    "fuck",
    "asshole",
    "idiot",
    "stupid",
    "fool",
    "loser",
    # Shortened/slang versions
    "bc",
    "mc",
    "wtf",
    "stfu",
    "gtfo",
    "mfr",
    "sob",
    "pos",
]


def detect_toxic(text):
    text = text.lower()
    return any(word in text for word in bad_words)


def add_toxicity(df):
    df["toxic"] = df["message"].apply(detect_toxic)
    return df
