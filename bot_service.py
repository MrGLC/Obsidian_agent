"""FastAPI service exposing the conversational bot and serving a basic UI."""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bot import process_query

app = FastAPI(title="Obsidian Bot Service")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    response = process_query(req.message)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
