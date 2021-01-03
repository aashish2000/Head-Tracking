import os
from starlette.responses import JSONResponse, StreamingResponse
from head_tracking import video_inference
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from asgiref.sync import sync_to_async

# Add Environment Variable for instructing the system to run inference on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = FastAPI()
app.mount("/video_files", StaticFiles(directory="./video_files"), name="video_files")

templates = Jinja2Templates(directory="templates")

# Display Home Webpage
@app.get("/", response_class=HTMLResponse)
async def display_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Upload and Play Videos
@app.post("/video_upload")
async def video_receive(request: Request):
    body = await request.form()
    video_name = "./video_files/uploaded_videos/" + body["fileToUpload"].filename
    contents = await body["fileToUpload"].read()

    with open(video_name,"wb") as f:
        f.write(contents)


# Process Uploaded Videos from Clients
@app.post("/process", response_class=HTMLResponse)
async def video_receive(request: Request):
    body = await request.form()
    video_name = "./video_files/uploaded_videos/"+body["file_name"]
    print(video_name)
    result_video, heads = await sync_to_async(video_inference)(video_name)
    return templates.TemplateResponse("show_result.html", {"request": request, "result_path": result_video, "head_count": heads})

# Download Processed Videos
@app.post("/download", response_class=FileResponse)
async def download_video(file_name: str = Form(...)):
    return FileResponse("./video_files/results/"+file_name, media_type='application/octet-stream', filename=file_name)    
