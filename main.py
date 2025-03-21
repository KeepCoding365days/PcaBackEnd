from fastapi import FastAPI,File,UploadFile
from pathlib import Path
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
from ai_helper import classify_images
from PIL import Image

origins = [
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

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict_grade/")
async def upload_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as f:
        f.write(await file.read())  # Save the file
    img = Image.open("test_image.jpg")
    result = classify_images(img)
    return {"grade": result}


