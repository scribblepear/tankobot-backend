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
        
        # Try to load pre-built vector database if it exists
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Check multiple possible locations for chroma_db
        # Look for both directory structures and direct sqlite files
        possible_paths = [
            "data/chroma_db",           # Standard directory structure
            "chroma_db",                # Root directory
            "data/chroma_db/chroma_db", # Nested structure
            "data"                      # Direct extraction to data/ (your case)
        ]
        
        chroma_dir = None
        for path in possible_paths:
            # Check if it's a directory with chroma files
            if os.path.exists(path) and os.path.isdir(path):
                # Look for chroma.sqlite3 or other chroma files
                chroma_files = [
                    os.path.join(path, "chroma.sqlite3"),
                    os.path.join(path, "index"),
                    os.path.join(path, "chroma.sqlite3-wal"),
                    os.path.join(path, "chroma.sqlite3-shm")
                ]
                if any(os.path.exists(f) for f in chroma_files):
                    chroma_dir = path
                    print(f"Found chroma_db at: {path}")
                    # Debug: show what files are in there
                    try:
                        files_found = [f for f in os.listdir(path) if 'chroma' in f.lower()]
                        print(f"Chroma files found: {files_found}")
                    except:
                        pass
                    break
        
        if api_key and chroma_dir:
            try:
                print(f"\nLoading pre-built vector database from: {chroma_dir}...")
                from langchain_openai import OpenAIEmbeddings
                from langchain_community.vectorstores import Chroma
                
                # Load embeddings function (needed for searching)
                embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model="text-embedding-3-small"
                )
                
                # Load the existing Chroma database
                db_mangas = Chroma(
                    persist_directory=chroma_dir,
                    embedding_function=embeddings
                )
                
                # Test it works
                test_results = db_mangas.similarity_search("test", k=1)
                print(f"✓ Vector database loaded successfully!")
                print(f"✓ Database contains pre-computed embeddings")
                print(f"✓ Test search returned {len(test_results)} results")
                
            except Exception as e:
                print(f"✗ Failed to load vector database: {e}")
                print(f"Attempting to debug the issue...")
                
                # Debug information
                try:
                    import glob
                    all_chroma_files = glob.glob("data/**/chroma*", recursive=True)
                    print(f"All chroma-related files found: {all_chroma_files}")
                    
                    if os.path.exists("data/chroma.sqlite3"):
                        file_size = os.path.getsize("data/chroma.sqlite3")
                        print(f"chroma.sqlite3 size: {file_size / 1024 / 1024:.2f} MB")
                except Exception as debug_e:
                    print(f"Debug failed: {debug_e}")
                
                db_mangas = None
        else:
            reasons = []
            if not api_key:
                reasons.append("no OpenAI API key")
            if not chroma_dir:
                reasons.append("chroma_db not found in any expected location")
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
