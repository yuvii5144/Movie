# 🎬 CineMatch AI — MP3 Movie Recommendation System

A software-only mini project using **Python + NLP + TF-IDF + Cosine Similarity**.

## 1. What the project does

The user selects a movie. The system reads its title, genres and description, converts the text into TF-IDF vectors, calculates cosine similarity against other movies, and displays the most similar movies.

## 2. Working

```text
Movie Dataset
     ↓
Data Cleaning
     ↓
Combine title + genre + overview
     ↓
NLP preprocessing
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity
     ↓
Top 6 Similar Movies
     ↓
Animated Recommendation UI
```

## 3. Installation on Windows

Open Command Prompt inside this project folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The browser should open automatically. If it does not, open the local address shown in the terminal.

## 4. Dataset

The app automatically downloads a public movie CSV on first run. For an offline setup, place your own compatible CSV at:

```text
data/movies.csv
```

The application automatically detects common column names such as:
- title / movie_title
- overview / description / plot
- genres / genre
- year / release_year
- rating / vote_average / imdb_rating

## 5. Viva explanation

### NLP
Natural Language Processing is used to process movie text such as genres and descriptions.

### TF-IDF
TF-IDF converts movie text into numerical vectors. Words that are useful for distinguishing movies receive higher importance.

### Cosine Similarity
Cosine similarity measures how similar two movie vectors are. A higher value means the movies have more similar text features.

### Example
If the user chooses a superhero action movie, movies with similar words, genres and descriptions receive higher similarity scores and appear first.

## 6. Advantages

- Easy to use
- No hardware
- Fast recommendation
- Simple ML/NLP pipeline
- Attractive interface
- Easy to explain in viva

## 7. Future scope

- User login and watch history
- Collaborative filtering
- Hybrid recommendation
- Real movie posters using an API
- User ratings
- Personalized recommendations
- Deep learning recommendation model
