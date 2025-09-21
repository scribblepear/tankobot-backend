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
    import requests
    os.makedirs("data", exist_ok=True)
    
    # UPDATE THESE URLs after uploading all files to GitHub Releases
    # Go to: https://github.com/scribblepear/tankobot/releases
    # Upload: mangas_cleaned.csv, mangas_with_emotions.csv, and tagged_description.txt
    files_to_download = {
        "data/mangas_cleaned.csv": "https://github.com/scribblepear/tankobot-backend/releases/download/v1.0/mangas_cleaned.csv",
        "data/mangas_with_emotions.csv": "https://github.com/scribblepear/tankobot-backend/releases/download/v1.0/mangas_with_emotions.csv",
        "data/tagged_description.txt": "https://github.com/scribblepear/tankobot-backend/releases/download/v1.0/tagged_description.txt"
    }
    
    for file_path, url in files_to_download.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path} from {url}...")
            try:
                # Try using curl or wget for better GitHub compatibility
                import subprocess
                
                # Try curl first (more commonly available)
                try:
                    result = subprocess.run(
                        ["curl", "-L", "-o", file_path, url],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    if result.returncode == 0:
                        print(f"Successfully downloaded with curl")
                    else:
                        raise Exception(f"Curl failed: {result.stderr}")
                except (FileNotFoundError, Exception) as e:
                    # Fallback to wget
                    try:
                        result = subprocess.run(
                            ["wget", "-O", file_path, url],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result.returncode == 0:
                            print(f"Successfully downloaded with wget")
                        else:
                            raise Exception(f"Wget failed: {result.stderr}")
                    except (FileNotFoundError, Exception) as e2:
                        # Final fallback to requests with different approach
                        print("Curl/wget not available, using requests...")
                        session = requests.Session()
                        session.headers.update({
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        
                        response = session.get(url, stream=True, allow_redirects=True)
                        response.raise_for_status()
                        
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                                if chunk:
                                    f.write(chunk)
                        print(f"Successfully downloaded with requests")
                
                # Verify file size
                file_size = os.path.getsize(file_path)
                print(f"File size: {file_size / 1024 / 1024:.2f} MB")
                
                # Verify text file content
                if file_path.endswith('.txt') and file_size < 1000:
                    print(f"WARNING: Text file seems too small ({file_size} bytes)")
                    # Delete the bad file so it re-downloads next time
                    os.remove(file_path)
                    raise Exception(f"Downloaded file too small, likely corrupted")
                        
            except Exception as e:
                print(f"Failed to download {file_path}: {e}")
                # Don't fail completely if tagged_description.txt fails
                if "tagged_description.txt" not in file_path:
                    return False
    
    # Verify critical files exist
    if not os.path.exists("data/mangas_with_emotions.csv"):
        print("Critical file mangas_with_emotions.csv missing!")
        return False
    
    return True

def initialize_database():
    """Initialize the vector database and load manga data"""
    global db_mangas, mangas_df, embeddings
    
    print("=" * 50)
    print("INITIALIZING DATABASE")
    print("=" * 50)
    
    # Try to download files if they don't exist
    if not os.path.exists("data/mangas_with_emotions.csv"):
        print("Data files not found locally, attempting to download...")
        if not download_data_files():
            print("Failed to download critical data files")
            return False
    
    try:
        # Load manga data
        print("Loading manga data...")
        mangas_df = pd.read_csv("data/mangas_with_emotions.csv")
        print(f"✓ Loaded {len(mangas_df)} manga records")
        
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
        
        # Try to initialize embeddings if API key and tagged_description.txt exist
        api_key = os.getenv("OPENAI_API_KEY")
        tagged_file = "data/tagged_description.txt"
        
        if api_key and os.path.exists(tagged_file) and os.path.getsize(tagged_file) > 0:
            try:
                print("\nAttempting to create embeddings...")
                print(f"OpenAI API key found: {api_key[:10]}...")
                
                # Read the entire file with explicit encoding
                with open(tagged_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                
                # Debug: Check what's actually in the file
                print(f"File size on disk: {os.path.getsize(tagged_file)} bytes")
                print(f"Content length: {len(content)} characters")
                print(f"First 500 chars: {repr(content[:500])}")
                
                # If file seems empty, try reading in binary mode
                if len(content) <= 1:
                    print("File appears empty, checking binary content...")
                    with open(tagged_file, "rb") as f:
                        binary_content = f.read()
                        print(f"Binary size: {len(binary_content)} bytes")
                        print(f"First 100 bytes: {binary_content[:100]}")
                    
                    # File is corrupted or not downloading properly
                    print("Tagged description file is not valid, skipping embeddings")
                    db_mangas = None
                    return  # Exit early
                
                # Parse the content into documents
                documents = []
                
                # Try splitting by newlines
                lines = content.split('\n')
                print(f"Split into {len(lines)} lines")
                
                for i, line in enumerate(lines[:500]):  # Process up to 500 lines
                    line = line.strip()
                    if line and len(line) > 20:
                        # Check if line starts with UID
                        import re
                        match = re.match(r'^(\d+)[:\s]+(.+)', line)
                        if match:
                            uid = match.group(1)
                            text = match.group(2)[:800]
                            doc_content = f"{uid}: {text}"
                        else:
                            doc_content = line[:800]
                            uid = str(i)
                        
                        doc = Document(
                            page_content=doc_content,
                            metadata={"source": "tagged_description.txt", "uid": uid}
                        )
                        documents.append(doc)
                
                print(f"Created {len(documents)} documents for embedding")
                
                if len(documents) > 0:
                    # Create embeddings
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    
                    # Test embeddings first
                    print("Testing embeddings...")
                    test_embedding = embeddings.embed_query("test manga search")
                    print(f"✓ Test embedding successful, dimension: {len(test_embedding)}")
                    
                    # Create vector store with documents
                    batch_size = min(100, len(documents))  # Start with up to 100 docs
                    print(f"Creating vector store with {batch_size} documents...")
                    
                    db_mangas = Chroma.from_documents(
                        documents[:batch_size],
                        embedding=embeddings,
                        ids=[str(i) for i in range(batch_size)]
                    )
                    print(f"✓ Vector store created with {batch_size} documents!")
                else:
                    print("No valid documents found for embeddings")
                    db_mangas = None
                    
            except Exception as e:
                print(f"✗ Embeddings initialization failed: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                print("Falling back to text search only")
                db_mangas = None
        else:
            reasons = []
            if not api_key:
                reasons.append("no OpenAI API key")
            if not os.path.exists(tagged_file):
                reasons.append("tagged_description.txt not found")
            elif os.path.getsize(tagged_file) == 0:
                reasons.append("tagged_description.txt is empty")
            print(f"Skipping embeddings ({', '.join(reasons)})")
            print("Will use text-based search instead")
            db_mangas = None
        
        print("\n" + "=" * 50)
        print("DATABASE INITIALIZATION COMPLETE")
        print(f"✓ Manga data loaded: {len(mangas_df)} records")
        print(f"✓ Search mode: {'Semantic (embeddings)' if db_mangas else 'Text-based'}")
        print("=" * 50 + "\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def retrieve_semantic_recommendations(
    query: str,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:
    """Retrieve manga recommendations using embeddings or text search"""
    
    if mangas_df is None or mangas_df.empty:
        return pd.DataFrame()
    
    # Try embeddings first if available
    if db_mangas is not None:
        try:
            print(f"Using semantic search for: {query}")
            # Perform similarity search
            recs = db_mangas.similarity_search(query, k=initial_top_k)
            manga_uids = []
            
            for rec in recs:
                content = rec.page_content.strip()
                # Extract UID from the content (should be at the start)
                match = re.match(r'^(\d+):', content)
                if match:
                    uid_num = int(match.group(1))
                    if uid_num in mangas_df["uid"].values:
                        manga_uids.append(uid_num)
            
            # Get manga details for found UIDs
            if manga_uids:
                # Preserve order of results
                manga_recs = mangas_df[mangas_df["uid"].isin(manga_uids)]
                # Sort by the order they appeared in search results
                manga_recs = manga_recs.set_index('uid').loc[manga_uids].reset_index()
                return manga_recs.head(final_top_k)
                
        except Exception as e:
            print(f"Semantic search failed, falling back to text search: {e}")
    
    # Fallback to text search
    print(f"Using text-based search for: {query}")
    try:
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 2]  # Filter short words
        
        # Score manga based on text matching
        scored_results = []
        
        # For performance, sample if dataset is large
        sample_size = min(15000, len(mangas_df))  # Check up to 15k manga
        if len(mangas_df) > sample_size:
            sample_df = mangas_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = mangas_df
        
        for idx, row in sample_df.iterrows():
            score = 0
            title = str(row.get('title', '')).lower()
            description = str(row.get('description', '')).lower()
            tags = str(row.get('tags', '')).lower()
            
            # Exact phrase match gets highest score
            if query_lower in title:
                score += 100
            elif query_lower in description:
                score += 50
            elif query_lower in tags:
                score += 30
            
            # Individual word matches
            for word in query_words:
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
        print("WARNING: Database initialization incomplete.")
        print("The API will still run but search functionality may be limited.")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Semantic Manga Recommender API",
        "database_loaded": mangas_df is not None and not mangas_df.empty,
        "search_mode": "semantic" if db_mangas is not None else "text",
        "total_manga": len(mangas_df) if mangas_df is not None else 0
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_manga(request: SearchRequest):
    """Search for manga using semantic or text search"""
    
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
        # Use the unified search function
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
            "database_loaded": False,
            "search_mode": "none"
        }
    
    return {
        "total_manga": len(mangas_df),
        "database_loaded": True,
        "search_mode": "semantic" if db_mangas is not None else "text",
        "available_tags": mangas_df["tags"].value_counts().head(20).to_dict() if "tags" in mangas_df.columns else {},
    }

@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check initialization status"""
    
    return {
        "database_loaded": db_mangas is not None,
        "dataframe_loaded": mangas_df is not None and not mangas_df.empty,
        "embeddings_loaded": embeddings is not None,
        "search_mode": "semantic" if db_mangas is not None else "text",
        "env_vars": {
            "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "NOT SET",
        },
        "files": {
            "data_dir_exists": os.path.exists("data"),
            "csv_exists": os.path.exists("data/mangas_with_emotions.csv"),
            "txt_exists": os.path.exists("data/tagged_description.txt"),
            "csv_size_mb": os.path.getsize("data/mangas_with_emotions.csv") / 1024 / 1024 if os.path.exists("data/mangas_with_emotions.csv") else 0,
            "txt_size_mb": os.path.getsize("data/tagged_description.txt") / 1024 / 1024 if os.path.exists("data/tagged_description.txt") else 0,
        },
        "dataframe_info": {
            "rows": len(mangas_df) if mangas_df is not None else 0,
            "columns": list(mangas_df.columns) if mangas_df is not None else []
        }
    }

@app.get("/debug/test-search/{query}")
async def test_search(query: str):
    """Test endpoint to debug search functionality"""
    
    if mangas_df is None:
        return {"error": "Database not loaded"}
    
    results = retrieve_semantic_recommendations(query, final_top_k=5)
    
    if results.empty:
        return {"query": query, "results": "No results found"}
    
    return {
        "query": query,
        "search_mode": "semantic" if db_mangas is not None else "text",
        "results": [
            {
                "title": row.get("title"),
                "uid": row.get("uid"),
                "description": str(row.get("description", ""))[:200]
            }
            for _, row in results.iterrows()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
