from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from ..ai.model import FakeModel

router = APIRouter()
prediction = FakeModel()

@router.post("/upload-image/")
async def create_upload_file(file: UploadFile = File(...)):
    result = prediction.predict(await file.read())  # Assuming predict method needs byte content
    return JSONResponse(content={"result": result})
