import streamlit as st
import zipfile
import os
import shutil
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Get some error to remove this i add this
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Page Title
st.set_page_config(page_title="WhatsApp Analyzer", layout="wide")

# Side Bar title
st.sidebar.title("📊 WhatsApp Chat Analyzer")

# File upload in Zip format
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp ZIP", type=["zip"])


def process_zip(uploaded_file):

    base_path = "chats"
    os.makedirs(base_path, exist_ok=True)

    # 👉 unique folder per upload
    chat_name = uploaded_file.name.split(".")[0].replace(" ", "_")
    chat_path = os.path.join(base_path, chat_name)

    if os.path.exists(chat_path):
        shutil.rmtree(chat_path)

    os.makedirs(chat_path)

    temp_path = os.path.join(chat_path, "temp")
    os.makedirs(temp_path)

    # extract
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall(temp_path)

    # media folders
    media_path = os.path.join(chat_path, "media")
    folders = {
        "images": os.path.join(media_path, "images"),
        "videos": os.path.join(media_path, "videos"),
        "audio": os.path.join(media_path, "audio"),
        "voice": os.path.join(media_path, "voice"),
    }

    for f in folders.values():
        os.makedirs(f, exist_ok=True)

    chat_txt_path = None

    # process files
    for file in os.listdir(temp_path):

        src = os.path.join(temp_path, file)

        if file.endswith(".txt"):
            chat_txt_path = os.path.join(chat_path, "chat.txt")
            shutil.move(src, chat_txt_path)

        elif file.startswith("IMG"):
            shutil.move(src, os.path.join(folders["images"], file))

        elif file.startswith("VID"):
            shutil.move(src, os.path.join(folders["videos"], file))

        elif file.startswith("AUD"):
            shutil.move(src, os.path.join(folders["audio"], file))

        elif file.startswith("PTT"):
            shutil.move(src, os.path.join(folders["voice"], file))

        else:
            pass  # ignore pdf, vcf, stk etc

    shutil.rmtree(temp_path)

    return chat_txt_path, chat_path


# model 1 time load hoga
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# Load Data
@st.cache_data
def load_data(uploaded_file):

    if uploaded_file.name.endswith(".zip"):

        chat_path, folder_path = process_zip(uploaded_file)

        with open(chat_path, "r", encoding="utf-8") as f:
            data = f.read()

    else:
        data = uploaded_file.getvalue().decode("utf-8")

    df = preprocessor.preprocess(data)

    return df, folder_path


# ---------------- MAIN ----------------
if uploaded_file is not None:

    df, folder_path = load_data(uploaded_file)

    # users
    user_list = df["user"].unique().tolist()
    if "group_notification" in user_list:
        user_list.remove("group_notification")

    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Select User", user_list)

    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    if st.sidebar.button("Analyze 🚀"):
        st.session_state.analysis_done = True

    if st.session_state.analysis_done:

        # ---------------- STATS ----------------
        st.title("📌 Top Statistics")

        (
            num_messages,
            words,
            total_media,
            actual_media,
            omitted_media,
            images,
            videos,
            audio,
            voice,
            num_links,
        ) = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages)
        col2.metric("Words", words)
        col3.metric("Links", num_links)
        col4.metric("Total Media", total_media)

        st.subheader("📁 Media Breakdown")

        col1, col2 = st.columns(2)
        col1.metric("Actual Files", actual_media)
        col2.metric("Omitted", omitted_media)

        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images", images)
        col2.metric("Videos", videos)
        col3.metric("Audio", audio)
        col4.metric("Voice Notes", voice)

        if actual_media == 0:
            st.warning("⚠️ This chat was exported without media.")

        st.divider()

        # ---------------- TIMELINE ----------------
        st.title("📈 Activity Timeline")

        col1, col2 = st.columns(2)

        with col1:
            timeline = helper.monthly_timeline(selected_user, df)
            fig = px.line(timeline, x="time", y="message")
            st.plotly_chart(fig, width="stretch")

        with col2:
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig = px.line(daily_timeline, x="only_date", y="message")
            st.plotly_chart(fig, width="stretch")

        st.divider()

        # -------------------- ACTIVITY MAP --------------------
        st.title("📅 Activity Insights")

        col1, col2 = st.columns(2)

        with col1:
            busy_day = helper.week_activity_map(selected_user, df)
            fig = px.bar(
                x=busy_day.index,
                y=busy_day.values,
                title="Most Busy Days",
                labels={"x": "Day", "y": "Messages"},
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            busy_month = helper.month_activity_map(selected_user, df)
            fig = px.bar(
                x=busy_month.index,
                y=busy_month.values,
                title="Most Busy Months",
                labels={"x": "Month", "y": "Messages"},
            )
            st.plotly_chart(fig, width="stretch")

        st.divider()

        # ---------------- HEATMAP ----------------
        st.subheader("🔥 Weekly Heatmap")

        user_heatmap = helper.activity_heatmap(selected_user, df)

        fig, ax = plt.subplots(figsize=(8, 6))  # control size here

        sns.heatmap(user_heatmap, ax=ax, cmap="YlGnBu", cbar=True)

        st.pyplot(fig, width="content")

        st.divider()

        # -------------------- MOST BUSY USERS --------------------
        if selected_user == "Overall":
            st.title("👥 Most Active Users")

            x, new_df = helper.most_busy_users(df)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    x=x.index,
                    y=x.values,
                    title="User Activity",
                    labels={"x": "User", "y": "Messages"},
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                st.dataframe(new_df, width="stretch")

        st.divider()

        # ---------------- WORDCLOUD ----------------
        st.title("☁️ WordCloud")

        df_wc = helper.create_wordcloud(selected_user, df)
        st.image(df_wc.to_array())

        st.divider()

        # ---------------- Most Common Words ----------------
        st.title("📝 Most Common Words")

        most_common_df = helper.most_common_words(selected_user, df)

        fig = px.bar(
            most_common_df,
            x="count",
            y="word",
            orientation="h",
            title="Top Words",
        )
        st.plotly_chart(fig, width="stretch")

        st.divider()

        # -------------------- EMOJI ANALYSIS --------------------
        st.title("😂 Emoji Analysis")

        emoji_df = helper.emoji_helper(selected_user, df)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df, width="stretch")

        with col2:
            fig = px.pie(
                emoji_df.head(),
                values=1,
                names=0,
                title="Top Emojis",
            )
            st.plotly_chart(fig, width="stretch")

        st.divider()

        # ---------------- IMAGES ----------------
        st.title("📸 Image Gallery")

        image_paths = helper.get_images(
            selected_user, df, f"{folder_path}/media/images"
        )

        if len(image_paths) == 0:
            st.info("📭 No images found in this chat.")
        else:
            cols = st.columns(4)
            for i, path in enumerate(image_paths):
                with cols[i % 4]:
                    st.image(path)

        st.divider()

        # ---------------- VIDEOS ----------------
        st.title("🎥 Video Preview")

        video_paths = helper.get_videos(
            selected_user, df, f"{folder_path}/media/videos"
        )

        if len(video_paths) == 0:
            st.info("📭 No videos found.")
        else:
            cols = st.columns(3)
            for i, path in enumerate(video_paths):
                with cols[i % 3]:
                    st.video(path)

        st.divider()

        # ---------------- VOICE ----------------
        st.title("🎙 Voice Notes")

        voice_paths = helper.get_voice_files(
            selected_user, df, f"{folder_path}/media/voice"
        )

        if len(voice_paths) == 0:
            st.info("📭 No voice notes available.")
        else:
            for path in voice_paths:
                st.audio(path)

        st.divider()

        # ---------------- AI INSIGHTS ----------------
        st.title("🤖 AI Insights")

        insights = helper.get_ai_insights(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Top User", insights["top_user"])
        col2.metric("Busiest Day", insights["busiest_day"])
        col3.metric("Top Media Sender", insights["media_user"])

        # -------------------- ML/AI FEATURES --------------------
        # Apply once
        df = helper.add_sentiment(df)
        df = helper.add_toxicity(df)

        # -------------------- Sentiment Analysis --------------------
        st.divider()

        st.title("😊 Sentiment Analysis")

        sent_counts = df["sentiment"].value_counts()

        fig = px.pie(
            names=sent_counts.index,
            values=sent_counts.values,
            title="Sentiment Distribution",
        )

        st.plotly_chart(fig, width="content")

        # -------------------- Similar Message --------------------
        st.divider()

        st.title("🔍 Similar Messages")

        @st.cache_data
        def get_embeddings_cached(df):
            return helper.get_embeddings(model, df)

        embeddings = get_embeddings_cached(df)

        if "results" not in st.session_state:
            st.session_state.results = None

        query = st.text_input("Enter message")

        if st.button("Search"):
            if query:
                st.session_state.results = helper.find_similar(
                    model, query, df, embeddings
                )

        # show result AFTER rerun
        if st.session_state.results:
            for r in st.session_state.results:
                st.write("👉", r)

        st.divider()

        # -------------------- Toxicity Detection --------------------
        st.title("🚨 Toxic Messages")

        toxic_df = df[df["toxic"] == True]

        st.write(f"Total Toxic Messages: {toxic_df.shape[0]}")

        st.dataframe(toxic_df[["user", "message"]].sample(10))

        st.divider()
