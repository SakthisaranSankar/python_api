from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong"})
