# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
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
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for storing loaded data
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

def initialize_database():
    """Initialize the vector database and load manga data"""
    global db_mangas, mangas_df, embeddings
    
    print("Initializing database...")
    
    # Check if required files exist
    if not os.path.exists("data/mangas_with_emotions.csv"):
        print("Warning: mangas_with_emotions.csv not found. Using fallback data.")
        # You can implement fallback logic here
        return False
    
    if not os.path.exists("data/tagged_description.txt"):
        print("Warning: tagged_description.txt not found.")
        return False
    
    try:
        # Load manga data
        mangas_df = pd.read_csv("data/mangas_with_emotions.csv")
        
        # Ensure uid column exists
        if "uid" not in mangas_df.columns:
            mangas_df["uid"] = mangas_df.index
        
        # Process cover images
        if "cover" in mangas_df.columns:
            mangas_df["large_cover"] = mangas_df["cover"].astype(str) + "&file=w800"
            mangas_df["large_cover"] = np.where(
                mangas_df["large_cover"].isna() | (mangas_df["large_cover"] == "nan&file=w800"),
                "/static/cover-not-found.jpg",
                mangas_df["large_cover"]
            )
        else:
            mangas_df["large_cover"] = "/static/cover-not-found.jpg"
        
        # Load and process documents
        raw_documents = TextLoader("data/tagged_description.txt").load()
        
        # Remove duplicates
        unique_texts = list({doc.page_content.strip() for doc in raw_documents})
        
        documents = [
            Document(page_content=text, metadata={"source": "tagged_description.txt"})
            for text in unique_texts
        ]
        
        # Split documents
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
        documents = text_splitter.split_documents(documents)
        
        # Assign IDs
        for i, doc in enumerate(documents):
            doc.metadata["id"] = str(i)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create vector store
        db_mangas = Chroma.from_documents(
            documents,
            embedding=embeddings,
            ids=[doc.metadata["id"] for doc in documents]
        )
        
        print("Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def retrieve_semantic_recommendations(
    query: str,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:
    """Retrieve manga recommendations based on semantic search"""
    
    if db_mangas is None or mangas_df is None:
        return pd.DataFrame()
    
    try:
        # Perform similarity search
        recs = db_mangas.similarity_search(query, k=initial_top_k)
        manga_list = []
        
        for rec in recs:
            content = rec.page_content.strip()
            # Extract UID from the content
            match = re.search(r'\d+', content)
            if match:
                uid_num = int(match.group())
                if uid_num in mangas_df["uid"].values:
                    manga_list.append(uid_num)
        
        # Filter mangas by matched UIDs
        manga_recs = mangas_df[mangas_df["uid"].isin(manga_list)].head(final_top_k)
        
        return manga_recs
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return pd.DataFrame()

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
    """Search for manga based on simple text matching"""
    
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
        query_lower = request.query.lower()
        query_words = query_lower.split()
        
        # Score each manga
        scored_results = []
        
        # Sample for performance - check 5000 random manga
        sample_size = min(5000, len(mangas_df))
        sample_df = mangas_df.sample(n=sample_size)
        
        for _, row in sample_df.iterrows():
            score = 0
            title = str(row.get('title', '')).lower()
            description = str(row.get('description', '')).lower()
            tags = str(row.get('tags', '')).lower()
            
            # Exact phrase match gets highest score
            if query_lower in title:
                score += 10
            if query_lower in description:
                score += 5
            if query_lower in tags:
                score += 3
            
            # Individual word matches
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    if word in title: score += 2
                    if word in description: score += 1
                    if word in tags: score += 1
            
            if score > 0:
                scored_results.append((score, row))
        
        # Sort by score and get top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_results[:request.limit or 16]
        
        # Convert to response format
        results = []
        for score, row in top_results:
            manga = MangaResponse(
                uid=int(row.get("uid", 0)),
                title=str(row.get("title", "Unknown Title")),
                description=str(row.get("description", "No description available"))[:500],
                cover=str(row.get("large_cover", "/static/cover-not-found.jpg")),
                rating=float(row.get("rating")) if pd.notna(row.get("rating")) else None,
                year=int(row.get("year")) if pd.notna(row.get("year")) else None,
                tags=str(row.get("tags", "")) if pd.notna(row.get("tags")) else ""
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

@app.get("/api/stats")
async def get_stats():
    """Get statistics about the database"""
    
    if mangas_df is None:
        return {
            "total_manga": 0,
            "database_loaded": False
        }
    
    return {
        "total_manga": len(mangas_df),
        "database_loaded": True,
        "available_tags": mangas_df["tags"].value_counts().head(10).to_dict() if "tags" in mangas_df.columns else {},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
