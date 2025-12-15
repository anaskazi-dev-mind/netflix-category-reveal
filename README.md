# 🎬 Netflix Category Reveal AI

### Uncovering Hidden Genres in Movie Data using Unsupervised Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://netflix-ai-portfolio.onrender.com)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

**[🔴 Live Demo: Click Here to Launch App](https://netflix-category-reveal.onrender.com)** 

---

## 📌 Project Overview
Netflix has thousands of titles, but standard genres like "Action" or "Comedy" are often too broad. This project uses **Natural Language Processing (NLP)** and **K-Means Clustering** to analyze over 8,000 movie plot descriptions and automatically group them into distinct, semantic categories (e.g., "Period Dramas," "Zombie Horror," "High School Rom-Coms") without any human supervision.

This demonstrates how AI can be used to improve content recommendation systems by understanding the *context* of a story, not just its tags.

## 🚀 Key Features
* **🤖 Unsupervised Learning:** Uses **K-Means Clustering** to find hidden patterns in text data.
* **📝 NLP Pipeline:** Implements **TF-IDF Vectorization** (with Lemmatization & Stopword removal) to convert text to machine-readable numbers.
* **📊 Dimensionality Reduction:** Uses **PCA (Principal Component Analysis)** to visualize high-dimensional clusters on a 2D interactive map.
* **🔎 Optimization:** Includes an "Elbow Method" analysis to mathematically determine the optimal number of categories ($k$).
* **🎨 Interactive Dashboard:** Built with **Streamlit** and **Plotly** for a professional, Netflix-themed UI.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (KMeans, PCA, TfidfVectorizer)
* **NLP:** NLTK (Stopwords, WordNetLemmatizer)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Deployment:** Render Cloud

---

## 🧠 How It Works (The Logic)
1.  **Data Cleaning:** The raw plot summaries are cleaned (lowercased, punctuation removed) and lemmatized (e.g., "running" → "run").
2.  **Vectorization:** We use **TF-IDF** to identify unique keywords. Common words like "the" are ignored, while rare, descriptive words like "vampire" or "detective" get higher weights.
3.  **Clustering:** The **K-Means** algorithm groups movies based on the similarity of their mathematical vectors.
4.  **Visualization:** Since the data has thousands of dimensions (one for each word), we use **PCA** to squash it down to X and Y coordinates so we can plot it on a graph.

---

## 📸 Screenshots

### 1. The Cluster Map (Interactive)
*Each dot is a movie. Colors represent the AI-discovered categories.*
![Cluster Map](https://github.com/anaskazi-dev-mind/netflix-category-reveal/blob/main/cluster_map.png?raw=true)

### 2. The Elbow Curve (Optimization)
*Mathematical proof of the optimal number of clusters.*
![Elbow Curve](https://github.com/anaskazi-dev-mind/netflix-category-reveal/blob/main/elbow_curve.png?raw=true)

---

## 💻 Local Installation

Want to run this on your own machine?

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/anaskazi-dev-mind/netflix-category-reveal.git](https://github.com/anaskazi-dev-mind/netflix-category-reveal.git)
    cd netflix-category-reveal
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

## 👤 Author
**Anas Kazi** *Computer Science & Engineering Student* [LinkedIn](https://linkedin.com/in/anaskazi001) | [GitHub](https://github.com/anaskazi-dev-mind)

---
*Created as a Portfolio Project for Data Science & AI Engineering.*
