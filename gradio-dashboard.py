from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import shutil
import os
import re
import requests

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

import gradio as gr

# -------------------------------
# Copy dataset locally
src = "/Users/jason/.cache/kagglehub/datasets/victorsoeiro/manga-manhwa-and-manhua-dataset/versions/1/data.csv"
dst = os.path.join(os.getcwd(), "mangas_with_emotions.csv")
shutil.copy(src, dst)

# Load mangas
mangas = pd.read_csv("mangas_with_emotions.csv")
mangas["large_cover"] = mangas["cover"] + "&file=w800"
mangas["large_cover"] = np.where(
    mangas["large_cover"].isna(),
    "cover-not-found.jpg",
    mangas["large_cover"],
)

# Ensure uid column exists
if "uid" not in mangas.columns:
    mangas["uid"] = mangas.index

# -------------------------------
# Load and split documents
raw_documents = TextLoader("tagged_description.txt").load()

unique_texts = list({doc.page_content.strip() for doc in raw_documents})

documents = [
    Document(page_content=text, metadata={"source": "tagged_description.txt"})
    for text in unique_texts
]

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
documents = text_splitter.split_documents(documents)

for i, doc in enumerate(documents):
    doc.metadata["id"] = str(i)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db_mangas = Chroma.from_documents(
    documents,
    embedding=embeddings,
    ids=[doc.metadata["id"] for doc in documents]
)

# -------------------------------
# Helper function to check if image URL is valid
def is_valid_image(url):
    try:
        response = requests.head(url, timeout=2)
        content_type = response.headers.get('Content-Type', '')
        return response.status_code == 200 and 'image' in content_type
    except:
        return False

# -------------------------------
# Semantic recommendation function
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_mangas.similarity_search(query, k=initial_top_k)
    manga_list = []

    for rec in recs:
        content = rec.page_content.strip()
        match = re.search(r'\d+', content)
        if match:
            uid_num = int(match.group())
            if uid_num in mangas["uid"].values:
                manga_list.append(uid_num)
            else:
                print(f"Warning: UID {uid_num} not found in mangas DataFrame.")
        else:
            print(f"Warning: No valid number found in rec: {content}")

    # Filter mangas by matched uids
    manga_recs = mangas[mangas["uid"].isin(manga_list)].head(final_top_k)

    # Apply category filter
    if category != "ALL":
        manga_recs = manga_recs[manga_recs["tags"] == category].head(final_top_k)

    # Apply tone sorting
    if tone == "Happy":
        manga_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        manga_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        manga_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        manga_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        manga_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return manga_recs

# -------------------------------
# Gradio recommendation function
def recommend_mangas(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        image_url = row["large_cover"]
        if not isinstance(image_url, str) or not is_valid_image(image_url):
            image_url = "cover-not-found.jpg"  # fallback

        description = row.get("description", "")
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        caption = f"{row.get('title', 'No Title')}: {truncated_description}"

        results.append([image_url, caption])  # format for gr.Gallery

    return results

# -------------------------------
# Gradio UI
tags = ["ALL"] + sorted(mangas["tags"].unique())
tones = ["ALL"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Manga/Manhua/Manwha Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a manga/manhua/manwha:",
            placeholder="e.g., A story about forgiveness"
        )
        tag_dropdown = gr.Dropdown(choices=tags, label="Select a category:", value="ALL")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="ALL")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommend Mangas/Manhuas/Manwhas", columns=8, rows=2)

    submit_button.click(fn=recommend_mangas,
                        inputs=[user_query, tag_dropdown, tone_dropdown],
                        outputs=output)

# -------------------------------
if __name__ == "__main__":
    dashboard.launch()
