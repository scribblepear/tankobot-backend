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

# Try different import methods for embeddings
try:
    from langchain_openai import OpenAIEmbeddings
    print("Using langchain_openai for embeddings")
except ImportError:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        print("Using langchain_community for embeddings")
    except ImportError:
        from langchain.embeddings import OpenAIEmbeddings
        print("Using langchain for embeddings")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Semantic Manga Recommender API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://scribblepear.github.io",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "*"  # Be careful with this in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600
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
    """Download data files from Google Drive if they don't exist"""
    os.makedirs("data", exist_ok=True)
    
    # Google Drive direct download links
    files_to_download = {
        "data/mangas_cleaned.csv": "https://drive.google.com/uc?export=download&id=1TbS-JUhEX2aExdL1ELIorbI5ARuDWitS",
        "data/mangas_with_emotions.csv": "https://drive.google.com/uc?export=download&id=1KdhLZYvoRtbrMV-wOqPGzqIAzJK7qDs2"
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
    
    # Move tagged_description.txt to data folder if it's in root
    if os.path.exists("tagged_description.txt") and not os.path.exists("data/tagged_description.txt"):
        print("Moving tagged_description.txt to data folder...")
        os.rename("tagged_description.txt", "data/tagged_description.txt")
    
    return True

def initialize_database():
    """Initialize the vector database and load manga data"""
    global db_mangas, mangas_df, embeddings, last_error
    
    print("Initializing database...")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        last_error = "OPENAI_API_KEY not found"
        return False
    else:
        print(f"OpenAI API key found: {api_key[:8]}...")
    
    # Check if required files exist
    if not os.path.exists("data/mangas_with_emotions.csv"):
        print("Warning: mangas_with_emotions.csv not found. Using fallback data.")
        last_error = "mangas_with_emotions.csv not found"
        return False
    
    if not os.path.exists("data/tagged_description.txt"):
        print("Warning: tagged_description.txt not found.")
        # Try to create a simple tagged_description.txt if it doesn't exist
        try:
            print("Creating tagged_description.txt from CSV data...")
            temp_df = pd.read_csv("data/mangas_with_emotions.csv")
            created_count = 0
            with open("data/tagged_description.txt", "w", encoding="utf-8") as f:
                for idx, row in temp_df.iterrows():
                    if pd.notna(row.get("description")) and pd.notna(row.get("uid")):
                        # Write each manga description on its own line
                        desc = str(row['description']).replace('\n', ' ').strip()[:500]
                        if desc:  # Only write non-empty descriptions
                            f.write(f"{int(row['uid'])}: {desc}\n")
                            created_count += 1
                    if created_count >= 1000:  # Limit to first 1000 for faster testing
                        break
            print(f"Successfully created tagged_description.txt with {created_count} entries")
        except Exception as e:
            print(f"Could not create tagged_description.txt: {e}")
            last_error = f"Could not create tagged_description.txt: {e}"
            return False
    
    # Verify the file has content
    try:
        with open("data/tagged_description.txt", "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                print("ERROR: tagged_description.txt is empty!")
                # Try to recreate it
                print("Attempting to recreate tagged_description.txt...")
                temp_df = pd.read_csv("data/mangas_with_emotions.csv")
                created_count = 0
                with open("data/tagged_description.txt", "w", encoding="utf-8") as f:
                    for idx, row in temp_df.head(1000).iterrows():  # Use first 1000 rows
                        if 'description' in row and pd.notna(row['description']):
                            uid = row.get('uid', idx)
                            desc = str(row['description']).replace('\n', ' ').strip()[:500]
                            if desc:
                                f.write(f"{uid}: {desc}\n")
                                created_count += 1
                print(f"Recreated with {created_count} entries")
    except Exception as e:
        print(f"Error checking tagged_description.txt: {e}")
    
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
            mangas_df["large_cover"] = mangas_df["cover"].astype(str) + "&file=w800"
            mangas_df["large_cover"] = np.where(
                mangas_df["large_cover"].isna() | (mangas_df["large_cover"] == "nan&file=w800"),
                "/static/cover-not-found.jpg",
                mangas_df["large_cover"]
            )
        else:
            mangas_df["large_cover"] = "/static/cover-not-found.jpg"
        
        # Load and process documents
        print("Loading text documents...")
        
        # Read the file directly to debug
        with open("data/tagged_description.txt", "r", encoding="utf-8") as f:
            file_content = f.read()
            print(f"File size: {len(file_content)} characters")
            lines = [line.strip() for line in file_content.split('\n') if line.strip()]
            print(f"Found {len(lines)} non-empty lines in file")
        
        if len(lines) == 0:
            print("ERROR: No content in tagged_description.txt!")
            last_error = "No content in tagged_description.txt"
            return False
        
        # Create documents from lines
        documents = []
        for line in lines[:1000]:  # Limit to 1000 for performance
            if line.strip():
                documents.append(
                    Document(page_content=line.strip(), metadata={"source": "tagged_description.txt"})
                )
        
        print(f"Created {len(documents)} documents")
        
        # Ensure we have documents to embed
        if len(documents) == 0:
            print("ERROR: No documents to embed!")
            last_error = "No documents to embed after processing"
            return False
        
        # Assign IDs
        for i, doc in enumerate(documents):
            doc.metadata["id"] = str(i)
        
        print(f"Ready to embed {len(documents)} documents")
        
        # Create embeddings with multiple fallback approaches
        print("Creating embeddings...")
        embeddings = None
        
        # Set environment variable explicitly
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Approach 1: Try the simplest initialization
        try:
            from langchain_openai.embeddings import OpenAIEmbeddings as OAIEmbed
            embeddings = OAIEmbed()
            print("Created embeddings using langchain_openai")
        except Exception as e:
            print(f"Approach 1 failed: {e}")
            last_error = f"Embeddings Approach 1 failed: {str(e)}"
        
        # Approach 2: Try with different import
        if embeddings is None:
            try:
                from langchain.embeddings.openai import OpenAIEmbeddings as OAIEmbed2
                embeddings = OAIEmbed2()
                print("Created embeddings using langchain.embeddings")
            except Exception as e:
                print(f"Approach 2 failed: {e}")
                last_error = f"Embeddings Approach 2 failed: {str(e)}"
        
        # Approach 3: Manual OpenAI client approach
        if embeddings is None:
            try:
                print("Trying manual OpenAI client approach...")
                import openai
                from langchain.embeddings.base import Embeddings
                from typing import List
                
                class SimpleOpenAIEmbeddings(Embeddings):
                    def __init__(self):
                        self.client = openai.OpenAI(api_key=api_key)
                    
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        embeddings = []
                        for text in texts:
                            response = self.client.embeddings.create(
                                model="text-embedding-3-small",
                                input=text
                            )
                            embeddings.append(response.data[0].embedding)
                        return embeddings
                    
                    def embed_query(self, text: str) -> List[float]:
                        response = self.client.embeddings.create(
                            model="text-embedding-3-small",
                            input=text
                        )
                        return response.data[0].embedding
                
                embeddings = SimpleOpenAIEmbeddings()
                print("Created custom embeddings wrapper")
            except Exception as e:
                print(f"Approach 3 failed: {e}")
                last_error = f"All embedding approaches failed. Last error: {str(e)}"
                
        if embeddings is None:
            raise Exception(f"Could not create OpenAI embeddings: {last_error}")
        
        # Create vector store
        print("Creating vector store...")
        db_mangas = Chroma.from_documents(
            documents,
            embedding=embeddings,
            ids=[doc.metadata["id"] for doc in documents]
        )
        print("Vector store created successfully!")
        
        print("Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        import traceback
        full_error = traceback.format_exc()
        print(f"Full traceback: {full_error}")
        last_error = f"Database init error: {str(e)}"
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    # Download files first
    print("Checking for data files...")
    download_success = download_data_files()
    if not download_success:
        print("Warning: Could not download all data files")
    
    # Then initialize database
    success = initialize_database()
    if not success:
        print("Warning: Database initialization incomplete. Some features may not work.")

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
    """
    Search for manga based on semantic similarity to the query
    """
    
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
            final_top_k=request.limit
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
            detail=f"An error occurred during search: {str(e)}"
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

@app.get("/debug/files")
async def debug_files():
    """Check what files exist in the deployment"""
    import os
    
    files_info = {
        "current_dir": os.getcwd(),
        "directory_contents": os.listdir("."),
        "data_dir_exists": os.path.exists("data"),
        "data_contents": os.listdir("data") if os.path.exists("data") else None,
        "csv_exists": os.path.exists("data/mangas_with_emotions.csv"),
        "txt_exists": os.path.exists("data/tagged_description.txt"),
        "env_vars": {
            "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "NOT SET",
            "PORT": os.getenv("PORT", "not set")
        }
    }
    return files_info

@app.get("/debug/check-data")
async def check_data():
    """Check the data files and their contents"""
    result = {
        "csv_exists": os.path.exists("data/mangas_with_emotions.csv"),
        "txt_exists": os.path.exists("data/tagged_description.txt")
    }
    
    if result["csv_exists"]:
        try:
            df = pd.read_csv("data/mangas_with_emotions.csv")
            result["csv_rows"] = len(df)
            result["csv_columns"] = list(df.columns)
            result["has_description_column"] = "description" in df.columns
            result["has_uid_column"] = "uid" in df.columns
            
            if "description" in df.columns:
                result["descriptions_with_content"] = df["description"].notna().sum()
                result["sample_description"] = str(df[df["description"].notna()]["description"].iloc[0])[:200] if result["descriptions_with_content"] > 0 else None
        except Exception as e:
            result["csv_error"] = str(e)
    
    if result["txt_exists"]:
        try:
            with open("data/tagged_description.txt", "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.strip().split('\n')
                result["txt_size"] = len(content)
                result["txt_lines"] = len(lines)
                result["txt_non_empty_lines"] = len([l for l in lines if l.strip()])
                result["txt_sample"] = lines[0][:200] if lines and lines[0] else None
        except Exception as e:
            result["txt_error"] = str(e)
    
    return result

# Global variable to store last error
last_error = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
