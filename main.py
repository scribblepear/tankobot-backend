# backend/main.py

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

def download_data_files():
    """Download data files from GitHub Releases"""
    import urllib.request
    os.makedirs("data", exist_ok=True)
    
    # UPDATE THESE URLs after you upload your files to GitHub Releases
    # Go to: https://github.com/scribblepear/tankobot/releases
    # Create a new release and upload your CSV files
    # Then replace these URLs with the actual download links
    files_to_download = {
        "data/mangas_cleaned.csv": "https://github.com/scribblepear/tankobot-backend/releases/download/v1.0/mangas_cleaned.csv",
        "data/mangas_with_emotions.csv": "https://github.com/scribblepear/tankobot-backend/releases/download/v1.0/mangas_with_emotions.csv"
    }
    
    for file_path, url in files_to_download.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path} from {url}...")
            try:
                # Download with progress indication
                def download_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    print(f"Progress: {percent:.1f}%", end='\r')
                
                urllib.request.urlretrieve(url, file_path, reporthook=download_progress)
                print(f"\nSuccessfully downloaded {file_path}")
                
                # Check file size to confirm download
                file_size = os.path.getsize(file_path)
                print(f"File size: {file_size / 1024 / 1024:.2f} MB")
            except Exception as e:
                print(f"Failed to download {file_path}: {e}")
                return False
    
    # Create tagged_description.txt from the CSV if it doesn't exist
    if not os.path.exists("data/tagged_description.txt"):
        print("Creating tagged_description.txt from CSV...")
        try:
            df = pd.read_csv("data/mangas_with_emotions.csv")
            with open("data/tagged_description.txt", "w", encoding="utf-8") as f:
                for idx, row in df.iterrows():
                    if pd.notna(row.get("description")):
                        uid = row.get('uid', idx)
                        title = row.get('title', '')
                        desc = str(row['description']).replace('\n', ' ').strip()[:1000]
                        tags = row.get('tags', '')
                        searchable_text = f"{uid}: {title} {desc} {tags}"
                        f.write(f"{searchable_text}\n")
            print("Created tagged_description.txt")
        except Exception as e:
            print(f"Could not create tagged_description.txt: {e}")
            # Continue anyway - we can still use text search
    
    return True

def initialize_database():
    """Initialize the vector database and load manga data"""
    global db_mangas, mangas_df, embeddings
    
    print("Initializing database...")
    
    # Try to download files if they don't exist
    if not os.path.exists("data/mangas_with_emotions.csv"):
        print("Data files not found locally, attempting to download...")
        if not download_data_files():
            print("Failed to download data files")
            return False
    
    try:
        # Load manga data
        print("Loading manga data...")
        mangas_df = pd.read_csv("data/mangas_with_emotions.csv")
        print(f"Loaded {len(mangas_df)} manga records")
        
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
        
        # Try to initialize embeddings if API key exists
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and os.path.exists("data/tagged_description.txt"):
            try:
                print("Attempting to create embeddings...")
                # Load and process documents
                raw_documents = TextLoader("data/tagged_description.txt").load()
                
                # Limit to first 100 documents for faster initialization
                unique_texts = list({doc.page_content.strip() for doc in raw_documents})[:100]
                
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
                
                # Create vector store with limited documents
                db_mangas = Chroma.from_documents(
                    documents[:50],  # Start with just 50 for faster startup
                    embedding=embeddings,
                    ids=[doc.metadata["id"] for doc in documents[:50]]
                )
                print("Embeddings created successfully!")
            except Exception as e:
                print(f"Embeddings initialization failed: {e}")
                print("Falling back to text search only")
                db_mangas = None
        else:
            print("Skipping embeddings (no API key or tagged_description.txt)")
            db_mangas = None
        
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
    """Retrieve manga recommendations using text search or embeddings"""
    
    if mangas_df is None or mangas_df.empty:
        return pd.DataFrame()
    
    # If we have embeddings, try to use them
    if db_mangas is not None:
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
            if manga_list:
                manga_recs = mangas_df[mangas_df["uid"].isin(manga_list)].head(final_top_k)
                if not manga_recs.empty:
                    return manga_recs
        except Exception as e:
            print(f"Embeddings search failed, falling back to text search: {e}")
    
    # Fallback to text search
    try:
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Score all manga (or a large sample for performance)
        scored_results = []
        
        # For performance, process in chunks or sample
        sample_size = min(10000, len(mangas_df))  # Check up to 10k manga
        if len(mangas_df) > sample_size:
            sample_df = mangas_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = mangas_df
        
        for idx, row in sample_df.iterrows():
            score = 0
            title = str(row.get('title', '')).lower()
            description = str(row.get('description', '')).lower()
            tags = str(row.get('tags', '')).lower()
            
            # Exact phrase gets highest score
            if query_lower in title:
                score += 100
            elif query_lower in description:
                score += 50
            elif query_lower in tags:
                score += 30
            
            # Individual word matches
            for word in query_words:
                if len(word) > 2:  # Skip short words
                    if word in title:
                        score += 10
                    if word in description:
                        score += 5
                    if word in tags:
                        score += 3
            
            if score > 0:
                scored_results.append((score, idx))
        
        # Sort by score and get top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scored_results[:final_top_k]]
        
        if top_indices:
            return mangas_df.loc[top_indices]
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in text search: {e}")
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
        "database_loaded": db_mangas is not None or (mangas_df is not None and not mangas_df.empty)
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_manga(request: SearchRequest):
    """Search for manga based on semantic or text matching"""
    
    if mangas_df is None or mangas_df.empty:
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
        # Use the retrieve_semantic_recommendations function which handles both embeddings and text search
        recommendations = retrieve_semantic_recommendations(
            query=request.query,
            final_top_k=request.limit or 16
        )
        
        if recommendations.empty:
            return SearchResponse(results=[], count=0)
        
        # Convert to response format
        results = []
        for _, row in recommendations.iterrows():
            # Process tags
            tags = row.get("tags", "")
            if pd.notna(tags) and tags:
                tags = str(tags).strip()
            else:
                tags = ""
            
            manga = MangaResponse(
                uid=int(row.get("uid", 0)),
                title=str(row.get("title", "Unknown Title")),
                description=str(row.get("description", "No description available"))[:500],
                cover=str(row.get("large_cover", "/static/cover-not-found.jpg")),
                rating=float(row.get("rating")) if pd.notna(row.get("rating")) else None,
                year=int(row.get("year")) if pd.notna(row.get("year")) else None,
                tags=tags
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

@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check initialization status"""
    import os
    
    return {
        "database_loaded": db_mangas is not None,
        "dataframe_loaded": mangas_df is not None and not mangas_df.empty,
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
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
