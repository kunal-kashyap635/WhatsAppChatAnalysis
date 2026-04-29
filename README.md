# 🚀 WhatsApp Chat Analyzer + AI Insights

A powerful **Streamlit-based WhatsApp Chat Analyzer** that not only visualizes chat statistics but also leverages **AI & Transformer models** to extract meaningful insights from conversations.

---

## 🌟 Features

### 📊 Chat Analytics

* Total messages, words, links, media stats
* Monthly & daily activity timeline
* Weekly activity heatmap
* Most active users
* WordCloud & most common words
* Emoji analysis

---

### 📸 Media Analysis

* Image gallery 📷
* Video preview 🎥
* Voice message playback 🎙️
* Handles:

  * With media export ✅
  * Without media export (graceful fallback) ✅

---

### 🤖 AI-Powered Features

#### 😊 Sentiment Analysis

* Classifies messages into:

  * Positive 😄
  * Negative 😡
  * Neutral 😐

---

#### 🔍 Semantic Search (Transformer-based)

* Uses **Sentence Transformers (BERT embeddings)**
* Find similar messages from chat history
* Context-aware search (not just keyword matching)

---

#### 🚨 Toxicity Detection

* Detects abusive or offensive messages
* Highlights toxic conversations

---

#### 😂 Funniest User Detection

* Based on:

  * Emoji usage
  * Slang words
* Finds the most entertaining person in the chat

---

## 🧠 Tech Stack

* **Frontend**: Streamlit
* **Data Processing**: Pandas, Regex
* **Visualization**: Plotly, Matplotlib, Seaborn
* **NLP / AI**:

  * Sentence Transformers (BERT)
  * VADER Sentiment Analysis
* **Others**:

  * WordCloud
  * Emoji processing

---

## ⚙️ How It Works

```text
Upload Chat (TXT / ZIP)
        ↓
Preprocessing (clean + structure)
        ↓
Analytics + Visualization
        ↓
AI Layer (Embeddings + Sentiment + Toxicity)
        ↓
Interactive Dashboard
```

---

## 📂 Project Structure

```
📁 whatsapp-chat-analyzer
│
├── app.py                # Streamlit UI
├── helper.py             # All analysis functions
├── preprocessor.py       # Chat parsing logic
├── stop_hinglish.txt     # Custom stopwords
├── chats/
│   └── media/            # Images, videos, audio
├── requirements.txt
└── README.md

```

---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

### 4️⃣ Upload Chat

* Export WhatsApp chat:

  * With media (recommended)
  * Or without media
* Upload `.txt` or `.zip`

---

## 💡 Key Highlights

* ⚡ Fast thanks to caching (`st.cache_resource`, `st.cache_data`)
* 🧠 Transformer-based semantic understanding
* 📱 Handles real-world messy WhatsApp data
* 🎯 Clean UX with fallback handling

---

## 🧪 Example Use Cases

* Analyze group dynamics
* Detect toxic conversations
* Find trending topics
* Discover most active or funniest user
* Search past conversations intelligently

---

## 🔮 Future Improvements

* 🗣 Voice-to-text (Whisper)
* 🤖 Smart reply suggestions
* 📸 Image captioning
* 📊 Topic modeling (BERTopic)
* 🌐 Deployment (Streamlit Cloud)

---

## 👨‍💻 Author

**Kunal Kashyap**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share with friends!

---

## 💬 Interview Insight

> “Built an AI-powered WhatsApp analyzer using transformer embeddings for semantic search, along with sentiment and behavioral insights.”

---

🔥 This project demonstrates real-world application of NLP + Data Analytics + UI engineering.
