import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# This exposes the FastAPI app to Vercel
handler = app
