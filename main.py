from fastapi import FastAPI,File,UploadFile
from pathlib import Path
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
from ai_helper import predict_grade
import uvicorn
origins = [
    '127.0.0.1',
    "*"  # Allow all origins (not recommended for production)
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allowed domains
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)
UPLOAD_DIR = Path("uploads")  # Create an "uploads" folder to store images
UPLOAD_DIR.mkdir(exist_ok=True)



@app.post("/predict_grade/")
async def upload_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    #print(file_path)
    with file_path.open("wb") as f:
        f.write(await file.read())  # Save the file
    print(file_path)
    grade=predict_grade(file_path)
    
    return {"grade":grade}




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 👈 use Render's PORT env var
    uvicorn.run("main:app", host="0.0.0.0", port=port)