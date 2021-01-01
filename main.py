import os
import pathlib
from head_tracking import video_inference
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
from asgiref.sync import sync_to_async

# Add Environment Variable for instructing the system to run inference on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Display Home Webpage
@app.get("/", response_class=HTMLResponse)
async def display_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Receive Videos from Clients
@app.post("/process", response_class=HTMLResponse)
async def video_receive(request: Request):
    body = await request.form()
    video_name = "./uploaded_videos/" + body["fileToUpload"].filename
    contents = await body["fileToUpload"].read()

    with open(video_name,"wb") as f:
        f.write(contents)
    
    result_video = await sync_to_async(video_inference)(video_name)
    return templates.TemplateResponse("show_result.html", {"request": request, "result_path": result_video})

@app.post("/download", response_class=FileResponse)
async def download_video(file_name: str = Form(...)):
    print("./results/"+file_name)

    return FileResponse("./results/"+file_name, media_type='application/octet-stream', filename=file_name)    



