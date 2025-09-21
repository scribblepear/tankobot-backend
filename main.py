from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import re
import urllib.request
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Semantic Manga Recommender API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins including GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db_mangas = None
mangas_df = None
embeddings = None

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 16

class MangaResponse(BaseModel):
    uid: int
    title: str
    description: str
    cover: str
    rating: Optional[float]
    year: Optional[int]
    tags: Optional[str]

class SearchResponse(BaseModel):
    results: List[MangaResponse]
    count: int

def download_data_files():
    """Download data files from Google Drive if they don't exist"""
    os.makedirs("data", exist_ok=True)
    
    files_to_download = {
        "data/mangas_cleaned.csv": "https://drive.google.com/uc?export=download&id=1TbS-JUhEX2aExdL1ELIorbI5ARuDWitS",
        "data/mangas_with_emotions.csv": "https://drive.google.com/uc?export=download&id=1KdhLZYvoRtbrMV-wOqPGzqIAzJK7qDs2",
        "data/tagged_description.txt": "https://drive.google.com/uc?export=download&id=1-GRSmhZZFRvz2_Lm0yH6omdIfce4MTwF"
    }
    
    for file_path, url in files_to_download.items():
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"Downloading {file_path}...")
            try:
                # Delete empty file if exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                urllib.request.urlretrieve(url, file_path)
                print(f"Successfully downloaded {file_path}")
            except Exception as e:
                print(f"Failed to download {file_path}: {e}")
                return False
    
    return True

def initialize_database():
    """Initialize the vector database and load manga data"""
    global db_mangas, mangas_df, embeddings
    
    print("Initializing database...")
    
    try:
        # Download files if needed
        download_success = download_data_files()
        if not download_success:
            print("Warning: Could not download all files")
        
        # Load manga data
        print("Loading manga CSV...")
        mangas_df = pd.read_csv("data/mangas_with_emotions.csv")
        print(f"Loaded {len(mangas_df)} manga records")
        
        # Ensure uid column exists
        if "uid" not in mangas_df.columns:
            mangas_df["uid"] = mangas_df.index
        
        # Process cover images
        if "cover" in mangas_df.columns:
            mangas_df["large_cover"] = mangas_df["cover"].apply(
                lambda x: str(x) + "&file=w800" if pd.notna(x) and str(x) != "nan" else "/static/cover-not-found.jpg"
            )
        else:
            mangas_df["large_cover"] = "/static/cover-not-found.jpg"
        
        # Check OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found - search functionality limited")
            return True  # Still return True so dataframe is available
        
        # Try to load and process documents for vector store
        if os.path.exists("data/tagged_description.txt"):
            print("Loading text documents...")
            try:
                with open("data/tagged_description.txt", "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                print(f"Found {len(lines)} documents")
                
                if len(lines) > 0:
                    # Create documents - limit for testing
                    documents = []
                    for line in lines[:50]:  # Start with just 50
                        if line.strip():
                            uid_match = re.match(r'^(\d+):', line)
                            if uid_match:
                                uid = uid_match.group(1)
                                doc = Document(
                                    page_content=line.strip(),
                                    metadata={"source": "tagged_description.txt", "uid": uid}
                                )
                                documents.append(doc)
                    
                    print(f"Created {len(documents)} documents for embedding")
                    
                    if len(documents) > 0:
                        # Try to create embeddings
                        try:
                            # Try different approaches
                            try:
                                from langchain_openai import OpenAIEmbeddings
                                embeddings = OpenAIEmbeddings(
                                    openai_api_key=api_key,
                                    model="text-embedding-3-small"
                                )
                                print("Created OpenAI embeddings")
                            except ImportError:
                                from langchain_community.embeddings import OpenAIEmbeddings
                                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                                print("Created community embeddings")
                            
                            # Create vector store
                            db_mangas = Chroma.from_documents(
                                documents[:10],  # Start with just 10
                                embedding=embeddings,
                                ids=[str(i) for i in range(min(10, len(documents)))]
                            )
                            print("Vector store created successfully!")
                            
                        except Exception as e:
                            print(f"Could not create embeddings: {e}")
                            print("Search will use fallback method")
                
            except Exception as e:
                print(f"Error processing documents: {e}")
        
        print("Database initialization complete!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def retrieve_semantic_recommendations(
    query: str,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:
    """Retrieve manga recommendations based on semantic search or fallback"""
    
    if mangas_df is None:
        return pd.DataFrame()
    
    # If vector database is available, use it
    if db_mangas is not None:
        try:
            # Perform similarity search
            recs = db_mangas.similarity_search(query, k=initial_top_k)
            manga_uids = []
            
            for doc in recs:
                # Try to get UID from metadata
                if 'uid' in doc.metadata:
                    uid = int(doc.metadata['uid'])
                    manga_uids.append(uid)
                else:
                    # Extract from content
                    content = doc.page_content.strip()
                    match = re.match(r'^(\d+):', content)
                    if match:
                        uid = int(match.group(1))
                        manga_uids.append(uid)
            
            # Get unique UIDs
            seen = set()
            unique_uids = []
            for uid in manga_uids:
                if uid not in seen and uid in mangas_df["uid"].values:
                    seen.add(uid)
                    unique_uids.append(uid)
            
            if unique_uids:
                # Return manga details
                manga_recs = mangas_df[mangas_df['uid'].isin(unique_uids)].head(final_top_k)
                return manga_recs
                
        except Exception as e:
            print(f"Error in semantic search: {e}")
    
    # Fallback: Simple text search on title and description
    print("Using fallback search method")
    query_lower = query.lower()
    query_words = query_lower.split()
    
    # Score each manga based on keyword matches
    scores = []
    for idx, row in mangas_df.iterrows():
        score = 0
        title = str(row.get('title', '')).lower()
        desc = str(row.get('description', '')).lower()
        tags = str(row.get('tags', '')).lower()
        
        for word in query_words:
            if word in title:
                score += 3  # Title matches are more important
            if word in desc:
                score += 1
            if word in tags:
                score += 2
        
        scores.append(score)
    
    mangas_df['search_score'] = scores
    results = mangas_df[mangas_df['search_score'] > 0].nlargest(final_top_k, 'search_score')
    
    # Clean up
    if 'search_score' in mangas_df.columns:
        mangas_df.drop('search_score', axis=1, inplace=True)
    
    return results

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    success = initialize_database()
    if not success:
        print("Warning: Database initialization incomplete. Some features may not work.")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Semantic Manga Recommender API",
        "database_loaded": db_mangas is not None
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_manga(request: SearchRequest):
    """Search for manga based on semantic similarity to the query"""
    
    if mangas_df is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized. Please try again later."
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Get recommendations
        recommendations = retrieve_semantic_recommendations(
            query=request.query,
            final_top_k=request.limit or 16
        )
        
        if recommendations.empty:
            return SearchResponse(results=[], count=0)
        
        # Convert to response format
        results = []
        for _, row in recommendations.iterrows():
            manga = MangaResponse(
                uid=int(row.get("uid", 0)),
                title=str(row.get("title", "Unknown Title")),
                description=str(row.get("description", "No description available"))[:500],
                cover=str(row.get("large_cover", "/static/cover-not-found.jpg")),
                rating=float(row.get("rating")) if pd.notna(row.get("rating")) else None,
                year=int(row.get("year")) if pd.notna(row.get("year")) else None,
                tags=str(row.get("tags", ""))
            )
            results.append(manga)
        
        return SearchResponse(
            results=results,
            count=len(results)
        )
        
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during search"
        )

@app.get("/api/manga/{uid}")
async def get_manga_details(uid: int):
    """Get detailed information about a specific manga"""
    
    if mangas_df is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized"
        )
    
    manga = mangas_df[mangas_df["uid"] == uid]
    
    if manga.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Manga with UID {uid} not found"
        )
    
    row = manga.iloc[0]
    
    return {
        "uid": int(row.get("uid", 0)),
        "title": str(row.get("title", "Unknown Title")),
        "description": str(row.get("description", "No description available")),
        "cover": str(row.get("large_cover", "/static/cover-not-found.jpg")),
        "rating": float(row.get("rating")) if pd.notna(row.get("rating")) else None,
        "year": int(row.get("year")) if pd.notna(row.get("year")) else None,
        "tags": str(row.get("tags", "")),
        "emotions": {
            "joy": float(row.get("happiness", 0)) if "happiness" in row else 0,
            "sadness": float(row.get("sadness", 0)) if "sadness" in row else 0,
            "anger": float(row.get("anger", 0)) if "anger" in row else 0,
            "fear": float(row.get("fear", 0)) if "fear" in row else 0,
            "surprise": float(row.get("surprise", 0)) if "surprise" in row else 0,
        }
    }

@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check initialization status"""
    import os
    
    return {
        "database_loaded": db_mangas is not None,
        "dataframe_loaded": mangas_df is not None,
        "embeddings_loaded": embeddings is not None,
        "env_vars": {
            "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "NOT SET",
        },
        "files": {
            "data_dir_exists": os.path.exists("data"),
            "csv_exists": os.path.exists("data/mangas_with_emotions.csv"),
            "txt_exists": os.path.exists("data/tagged_description.txt"),
        },
        "dataframe_info": {
            "rows": len(mangas_df) if mangas_df is not None else 0,
            "columns": list(mangas_df.columns) if mangas_df is not None else []
        } if mangas_df is not None else {}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
