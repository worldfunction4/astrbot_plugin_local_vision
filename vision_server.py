"""
本地识图服务
接收图片文件，调用 Ollama llava 模型进行识别，返回描述文本

使用方法:
    pip install fastapi uvicorn httpx
    python vision_server.py
"""
import logging
import tempfile
import os
import httpx
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llava:7b"
DEFAULT_PROMPT = "请详细描述这张图片的内容，用中文回答。"


async def check_model_exists(model: str) -> bool:
    """检测 Ollama 模型是否已安装"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/show",
                json={"name": model}
            )
            return resp.status_code == 200
    except httpx.RequestError:
        return False


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    model: str = Form(DEFAULT_MODEL),
):
    logger.info(f"Received image: {file.filename}, content_type: {file.content_type}, model: {model}")
    image_data = await file.read()

    # 检测模型是否存在
    if not await check_model_exists(model):
        error_msg = f"模型 '{model}' 未在本地找到，请先执行: ollama pull {model}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=422,
            content={
                "error": "model_not_found",
                "model": model,
                "message": error_msg
            }
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        logger.info(f"Saved to temp file: {tmp_path}, size: {len(image_data)} bytes")

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        logger.info(f"Calling Ollama with model: {model}")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)

        if resp.status_code != 200:
            error_msg = f"Ollama API error (status code: {resp.status_code})"
            logger.error(f"Ollama error: {error_msg}")
            return JSONResponse(status_code=500, content={"error": error_msg})

        result = resp.json()
        description = result["message"]["content"]
        logger.info(f"Got description, length: {len(description)}")
        return {"description": description}

    except Exception as e:
        logger.exception(f"Error analyzing image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temp file: {tmp_path}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
