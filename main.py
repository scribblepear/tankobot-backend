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
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Successfully downloaded {file_path}")
            except Exception as e:
                print(f"Failed to download {file_path}: {e}")
                return False
    
    return True

def create_embeddings_wrapper(api_key):
    """Create a simple embeddings wrapper that works with urllib"""
    from typing import List
    
    class SimpleEmbeddings:
        def __init__(self, api_key):
            self.api_key = api_key
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        
        def _get_embedding(self, text: str) -> List[float]:
            import json
            import time
            
            url = "https://api.openai.com/v1/embeddings"
            data = {
                "model": "text-embedding-3-small",
                "input": text[:8000]  # Limit text length
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=self.headers
            )
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=30) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        return result['data'][0]['embedding']
                except urllib.error.HTTPError as e:
                    if e.code in [503, 429] and retry < max_retries - 1:
                        wait_time = (retry + 1) * 2
                        print(f"API error {e.code}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            import time
            embeddings = []
            for i, text in enumerate(texts):
                if i > 0 and i % 10 == 0:
                    print(f"Embedding document {i}/{len(texts)}...")
                    time.sleep(0.5)  # Rate limiting
                embeddings.append(self._get_embedding(text))
            return embeddings
        
        def embed_query(self, text: str) -> List[float]:
            return self._get_embedding(text)
    
    return SimpleEmbeddings(api_key)

def initialize_database():
    """Initialize the vector database and load manga data"""
    global db_mangas, mangas_df, embeddings
    
    print("Initializing database...")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found!")
        return False
    else:
        print(f"OpenAI API key found: {api_key[:10]}...")
    
    # Download files if needed
    download_data_files()
    
    # Check if required files exist
    if not os.path.exists("data/mangas_with_emotions.csv"):
        print("ERROR: mangas_with_emotions.csv not found")
        return False
    
    # Check if tagged_description.txt exists and has content
    txt_needs_download = False
    if os.path.exists("data/tagged_description.txt"):
        with open("data/tagged_description.txt", "r", encoding="utf-8") as f:
            content = f.read()
            if len(content.strip()) == 0:
                print("Tagged description file is empty, deleting and redownloading...")
                os.remove("data/tagged_description.txt")
                txt_needs_download = True
    else:
        txt_needs_download = True
    
    # Download the file if needed
    if txt_needs_download:
        print("Downloading tagged_description.txt from Google Drive...")
        try:
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?export=download&id=1-GRSmhZZFRvz2_Lm0yH6omdIfce4MTwF",
                "data/tagged_description.txt"
            )
            print("Successfully downloaded tagged_description.txt")
        except Exception as e:
            print(f"Failed to download tagged_description.txt: {e}")
            # Create it from CSV as fallback
            print("Creating tagged_description.txt from CSV as fallback...")
            try:
                temp_df = pd.read_csv("data/mangas_with_emotions.csv")
                with open("data/tagged_description.txt", "w", encoding="utf-8") as f:
                    created_count = 0
                    for idx, row in temp_df.iterrows():
                        if pd.notna(row.get("description")) and str(row.get("description")).strip():
                            uid = row.get('uid', idx)
                            title = str(row.get('title', '')).strip()
                            desc = str(row['description']).replace('\n', ' ').strip()[:500]
                            tags = str(row.get('tags', '')).strip()
                            
                            if desc:
                                searchable_text = f"{uid}: {title} {desc} {tags}"
                                f.write(f"{searchable_text}\n")
                                created_count += 1
                                
                            if created_count >= 100:
                                break
                print(f"Created fallback tagged_description.txt with {created_count} entries")
            except Exception as e2:
                print(f"Could not create fallback: {e2}")
                return False
    
    try:
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
        
        # Load documents for vector store
        print("Loading text documents...")
        with open("data/tagged_description.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Found {len(lines)} documents")
        
        # Create documents - LIMIT TO 50 FOR TESTING
        documents = []
        for line in lines[:50]:  # Start with just 50 documents
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
        
        if len(documents) == 0:
            print("ERROR: No documents to embed!")
            return False
        
        # Try different embedding approaches
        print("Attempting to create embeddings...")
        
        # First try langchain-openai
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-3-small"
            )
            print("Created embeddings with langchain-openai")
        except Exception as e:
            print(f"langchain-openai failed: {e}")
            
            # Try langchain-community
            try:
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                print("Created embeddings with langchain-community")
            except Exception as e2:
                print(f"langchain-community failed: {e2}")
                
                # Fallback to custom wrapper
                try:
                    print("Using custom embeddings wrapper...")
                    embeddings = create_embeddings_wrapper(api_key)
                    print("Created custom embeddings wrapper")
                except Exception as e3:
                    print(f"Custom wrapper failed: {e3}")
                    return False
        
        # Create vector store with minimal documents
        print("Creating vector store...")
        try:
            # Start with just 10 documents to test
            test_docs = documents[:10]
            db_mangas = Chroma.from_documents(
                test_docs,
                embedding=embeddings,
                ids=[str(i) for i in range(len(test_docs))]
            )
            print(f"Created vector store with {len(test_docs)} documents")
            
            # If successful, add more documents in small batches
            if len(documents) > 10:
                for i in range(10, min(len(documents), 50), 10):
                    batch = documents[i:i+10]
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    ids = [str(j) for j in range(i, i+len(batch))]
                    
                    print(f"Adding batch {i//10}...")
                    try:
                        db_mangas.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                    except Exception as e:
                        print(f"Failed to add batch: {e}")
                        break
            
            print("Vector store created successfully!")
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
        
        print("Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
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
        # Perform similarity search with metadata
        recs = db_mangas.similarity_search_with_score(query, k=initial_top_k)
        manga_uids = []
        
        for doc, score in recs:
            # First try to get UID from metadata
            if 'uid' in doc.metadata:
                uid = int(doc.metadata['uid'])
                manga_uids.append(uid)
            else:
                # Fallback: extract from content
                content = doc.page_content.strip()
                match = re.match(r'^(\d+):', content)
                if match:
                    uid = int(match.group(1))
                    manga_uids.append(uid)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_uids = []
        for uid in manga_uids:
            if uid not in seen and uid in mangas_df["uid"].values:
                seen.add(uid)
                unique_uids.append(uid)
        
        # Get manga details for the UIDs
        if unique_uids:
            # Preserve the order of results by using categorical
            mangas_df['uid_cat'] = pd.Categorical(mangas_df['uid'], categories=unique_uids, ordered=True)
            manga_recs = mangas_df[mangas_df['uid'].isin(unique_uids)].sort_values('uid_cat')
            manga_recs = manga_recs.head(final_top_k).drop('uid_cat', axis=1)
            return manga_recs
        else:
            return pd.DataFrame()
        
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
    """Search for manga based on semantic similarity to the query"""
    
    if db_mangas is None or mangas_df is None:
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
            # Process tags
            tags = row.get("tags", "")
            if pd.notna(tags) and tags:
                # Clean up tags string
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
        }
    }

@app.get("/debug/download-txt")
async def download_txt():
    """Force download tagged_description.txt from Google Drive"""
    try:
        # Delete existing file if it exists
        if os.path.exists("data/tagged_description.txt"):
            os.remove("data/tagged_description.txt")
            
        # Download from Google Drive
        urllib.request.urlretrieve(
            "https://drive.google.com/uc?export=download&id=1-GRSmhZZFRvz2_Lm0yH6omdIfce4MTwF",
            "data/tagged_description.txt"
        )
        
        # Verify content
        with open("data/tagged_description.txt", "r", encoding="utf-8") as f:
            content = f.read()
            lines = [l for l in content.strip().split('\n') if l.strip()]
        
        return {
            "success": True,
            "file_size": len(content),
            "lines_in_file": len(lines),
            "sample": lines[0][:200] if lines else None
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/recreate-txt")
async def recreate_txt_file():
    """Manually recreate the tagged_description.txt file"""
    try:
        if not os.path.exists("data/mangas_with_emotions.csv"):
            return {"error": "CSV file not found"}
        
        temp_df = pd.read_csv("data/mangas_with_emotions.csv")
        
        # Delete existing file if it exists
        if os.path.exists("data/tagged_description.txt"):
            os.remove("data/tagged_description.txt")
        
        # Create new file
        with open("data/tagged_description.txt", "w", encoding="utf-8") as f:
            created_count = 0
            for idx, row in temp_df.iterrows():
                if pd.notna(row.get("description")) and str(row.get("description")).strip():
                    uid = row.get('uid', idx)
                    title = str(row.get('title', '')).strip()
                    desc = str(row['description']).replace('\n', ' ').strip()[:500]
                    tags = str(row.get('tags', '')).strip()
                    
                    if desc:
                        searchable_text = f"{uid}: {title} {desc} {tags}"
                        f.write(f"{searchable_text}\n")
                        created_count += 1
                        
                    if created_count >= 100:
                        break
        
        # Verify content
        with open("data/tagged_description.txt", "r") as f:
            content = f.read()
            lines = content.strip().split('\n')
        
        return {
            "success": True,
            "entries_created": created_count,
            "file_size": len(content),
            "lines_in_file": len(lines),
            "sample": lines[0][:200] if lines else None
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/reinit")
async def reinitialize():
    """Manually trigger reinitialization"""
    success = initialize_database()
    return {
        "success": success,
        "database_loaded": db_mangas is not None,
        "dataframe_loaded": mangas_df is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
