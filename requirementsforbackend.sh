# backend/requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.25.2
python-dotenv==1.0.0
langchain==0.0.340
langchain-community==0.0.10
langchain-openai==0.0.5
chromadb==0.4.18
openai==1.3.5
pydantic==2.5.0
python-multipart==0.0.6

# backend/.env (example file - create this with your actual API key)
OPENAI_API_KEY=your_openai_api_key_here

# backend/setup.sh
#!/bin/bash

# Create backend directory structure
mkdir -p backend/data
mkdir -p backend/static

# Create Python virtual environment
cd backend
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data
mkdir -p static

echo "Setup complete! Now:"
echo "1. Copy your 'tagged_description.txt' to backend/data/"
echo "2. Copy your 'mangas_with_emotions.csv' to backend/data/"
echo "3. Add your OpenAI API key to backend/.env"
echo "4. Run: python main.py"