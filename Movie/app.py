import os
import re
import html
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_DIR="data"
DATA_FILE=os.path.join(DATA_DIR,"movies.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# Public dataset URL. The app also works with a local data/movies.csv.
DATA_URL="https://raw.githubusercontent.com/selva86/datasets/master/movies.csv"

# ---------------- CSS / animated UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 15% 10%, rgba(124,58,237,.18), transparent 28%),
      radial-gradient(circle at 85% 15%, rgba(6,182,212,.14), transparent 25%),
      linear-gradient(135deg,#070b17 0%,#0b1020 48%,#080b16 100%);
    color:#eef2ff;
}
.hero {
    min-height: 270px;
    padding: 55px 35px 42px;
    border-radius: 28px;
    margin-bottom: 25px;
    background: linear-gradient(120deg,rgba(124,58,237,.32),rgba(6,182,212,.17),rgba(15,23,42,.8));
    border:1px solid rgba(148,163,184,.16);
    box-shadow:0 20px 70px rgba(0,0,0,.32);
    position:relative;
    overflow:hidden;
}
.hero:before,.hero:after{
    content:""; position:absolute; border-radius:50%; filter:blur(2px);
    animation:float 7s ease-in-out infinite;
}
.hero:before{width:180px;height:180px;right:8%;top:-60px;background:rgba(124,58,237,.18);}
.hero:after{width:120px;height:120px;right:25%;bottom:-65px;background:rgba(6,182,212,.13);animation-delay:1.5s;}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(18px)}}
.hero h1{font-size:56px!important;margin:0;font-weight:800;letter-spacing:-2px;}
.hero p{font-size:18px;color:#b9c4dc;margin-top:12px;max-width:760px;}
.badge{display:inline-block;padding:7px 12px;border-radius:999px;background:rgba(124,58,237,.18);border:1px solid rgba(167,139,250,.3);color:#c4b5fd;font-size:12px;font-weight:700;letter-spacing:1px;}
.section-title{font-size:24px;font-weight:800;margin:24px 0 12px;}
.card{
    background:linear-gradient(145deg,rgba(20,28,49,.94),rgba(10,16,30,.94));
    border:1px solid rgba(148,163,184,.14);
    border-radius:20px;padding:18px;
    min-height:150px;
    transition:.25s;
    box-shadow:0 10px 35px rgba(0,0,0,.18);
}
.card:hover{transform:translateY(-5px);border-color:rgba(124,58,237,.45);box-shadow:0 18px 50px rgba(76,29,149,.2);}
.movie-title{font-size:18px;font-weight:800;color:#f8fafc;margin-bottom:8px;}
.meta{font-size:13px;color:#94a3b8;line-height:1.55;}
.score{color:#67e8f9;font-weight:800;}
.poster{
    height:210px;border-radius:15px;
    display:flex;align-items:center;justify-content:center;text-align:center;
    padding:18px;font-size:23px;font-weight:800;color:white;
    background:linear-gradient(135deg,#312e81,#0e7490);
    box-shadow:inset 0 0 40px rgba(0,0,0,.2);
}
.stButton>button{
    border-radius:12px!important;border:1px solid rgba(167,139,250,.3)!important;
    background:linear-gradient(90deg,#6d28d9,#0891b2)!important;
    color:white!important;font-weight:700!important;
}
div[data-testid="stMetric"]{background:rgba(15,23,42,.65);padding:12px;border-radius:15px;border:1px solid rgba(148,163,184,.12);}
.footer{text-align:center;color:#64748b;padding:35px 0 10px;font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ---------------- Data ----------------
@st.cache_data(show_spinner=False)
def load_movies():
    if not os.path.exists(DATA_FILE):
        try:
            r=requests.get(DATA_URL,timeout=20)
            r.raise_for_status()
            with open(DATA_FILE,"wb") as f: f.write(r.content)
        except Exception:
            return pd.DataFrame()
    try:
        df=pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame()
    return df

def find_columns(df):
    lower={c.lower().strip():c for c in df.columns}
    title=next((lower[k] for k in ["title","movie_title","name"] if k in lower), None)
    overview=next((lower[k] for k in ["overview","plot","description","synopsis"] if k in lower), None)
    genre=next((lower[k] for k in ["genres","genre","category"] if k in lower), None)
    year=next((lower[k] for k in ["year","release_year","release_date"] if k in lower), None)
    rating=next((lower[k] for k in ["rating","vote_average","imdb_rating"] if k in lower), None)
    return title,overview,genre,year,rating

def prepare(df):
    title,overview,genre,year,rating=find_columns(df)
    if not title: return pd.DataFrame()
    out=pd.DataFrame()
    out["title"]=df[title].fillna("").astype(str)
    out["overview"]=df[overview].fillna("").astype(str) if overview else ""
    out["genre"]=df[genre].fillna("").astype(str) if genre else ""
    out["year"]=df[year].fillna("").astype(str) if year else ""
    out["rating"]=pd.to_numeric(df[rating],errors="coerce").fillna(0) if rating else 0
    out["text"]=(out["title"]+" "+out["genre"]+" "+out["overview"]).map(
        lambda x: re.sub(r"[^a-zA-Z0-9 ]"," ",x.lower())
    )
    out=out.drop_duplicates("title").reset_index(drop=True)
    return out

@st.cache_resource(show_spinner=False)
def build_model(texts):
    vec=TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1,2))
    matrix=vec.fit_transform(texts)
    return vec,matrix

def recommend(df, matrix, idx, n=6):
    sims=cosine_similarity(matrix[idx],matrix).ravel()
    order=np.argsort(sims)[::-1]
    order=[i for i in order if i!=idx][:n]
    return [(i,float(sims[i])) for i in order]

def poster_html(title, seed):
    palettes=[
        ("#312e81","#0891b2"),("#7c2d12","#be123c"),("#14532d","#0f766e"),
        ("#581c87","#1d4ed8"),("#164e63","#4f46e5"),("#713f12","#9f1239")
    ]
    a,b=palettes[seed%len(palettes)]
    safe=html.escape(title[:34])
    return f'<div class="poster" style="background:linear-gradient(135deg,{a},{b});">{safe}</div>'

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
<span class="badge">AI • NLP • TF-IDF • COSINE SIMILARITY</span>
<h1>🎬 CineMatch AI</h1>
<p>Discover your next movie using Natural Language Processing. Select a movie and let the AI find titles with similar stories, genres and descriptions.</p>
</div>
""",unsafe_allow_html=True)

df=prepare(load_movies())

if df.empty:
    st.error("Movie dataset could not be loaded. Put a CSV file named data/movies.csv in the project folder and restart the app.")
    st.stop()

vec,matrix=build_model(df["text"].tolist())

# ---------------- Search / selection ----------------
st.markdown('<div class="section-title">🔎 Find a Movie</div>',unsafe_allow_html=True)
c1,c2=st.columns([3,1])
with c1:
    query=st.text_input("Search",placeholder="Type a movie name...",label_visibility="collapsed")
with c2:
    st.metric("Movies in AI database",f"{len(df):,}")

filtered=df
if query.strip():
    q=query.lower()
    filtered=df[df["title"].str.lower().str.contains(re.escape(q),na=False)].head(20)

if len(filtered)==0:
    st.warning("No movie found. Try another title.")
    st.stop()

options=filtered["title"].tolist()
selected=st.selectbox("Choose a movie",options,index=0)
idx=int(df.index[df["title"].eq(selected)][0])

# ---------------- Selected movie ----------------
m=df.loc[idx]
st.markdown('<div class="section-title">🎞️ Selected Movie</div>',unsafe_allow_html=True)
a,b=st.columns([1,2])
with a:
    st.markdown(poster_html(m["title"],idx),unsafe_allow_html=True)
with b:
    st.markdown(f'<div class="card"><div class="movie-title">{html.escape(m["title"])}</div>'
                f'<div class="meta">Genre: <b>{html.escape(str(m["genre"]))}</b><br>'
                f'Year: <b>{html.escape(str(m["year"]))}</b><br>'
                f'Rating: <b>⭐ {float(m["rating"]):.1f}</b></div>'
                f'<br><div class="meta">{html.escape(str(m["overview"]))}</div></div>',unsafe_allow_html=True)

# ---------------- Recommendations ----------------
st.markdown('<div class="section-title">✨ AI Recommendations</div>',unsafe_allow_html=True)
recs=recommend(df,matrix,idx,6)

cols=st.columns(3)
for pos,(i,score) in enumerate(recs):
    x=df.loc[i]
    with cols[pos%3]:
        st.markdown(
            f'<div class="card">{poster_html(x["title"],i)}'
            f'<div style="height:12px"></div>'
            f'<div class="movie-title">{html.escape(x["title"])}</div>'
            f'<div class="meta">{html.escape(str(x["genre"]))}<br>'
            f'⭐ {float(x["rating"]):.1f} &nbsp; • &nbsp; '
            f'<span class="score">{score*100:.1f}% similarity</span></div></div>',
            unsafe_allow_html=True
        )

# ---------------- How it works ----------------
st.markdown('<div class="section-title">🧠 How CineMatch AI Works</div>',unsafe_allow_html=True)
steps=[
("01","Movie Data","Title + genre + overview"),
("02","NLP","Clean and normalize text"),
("03","TF-IDF","Convert text into numerical features"),
("04","Similarity","Compare movies using cosine similarity"),
("05","Recommendation","Show the most similar movies")
]
cc=st.columns(5)
for c,(n,t,d) in zip(cc,steps):
    with c:
        st.markdown(f'<div class="card"><div style="color:#67e8f9;font-size:13px;font-weight:800">{n}</div>'
                    f'<div class="movie-title" style="margin-top:8px">{t}</div>'
                    f'<div class="meta">{d}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="footer">CineMatch AI • Mini Project • Python + NLP • No hardware required</div>',unsafe_allow_html=True)
