import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="Stock Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

class StockAnalysisRequest(BaseModel):
    stock_code: str

@app.post("/api/analyze")
async def analyze_stock(request: StockAnalysisRequest):
    code = request.stock_code.strip()
    if not (code.isdigit() and 1 <= len(code) <= 6):
        raise HTTPException(status_code=400, detail="请输入1-6位数字股票代码")

    prompt = f"""
作为专业股票分析师，请对{code}进行研判。
输出严格JSON格式：
{{"sector":"行业","trend":"inflow/outflow","flow":"0.0","marketEmotion":"亢奋期|震荡期|冷静期","recommendation":"建议"}}
"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                QWEN_API_URL,
                headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "qwen-max",
                    "input": {"messages": [{"role": "user", "content": prompt}]},
                    "parameters": {"temperature": 0.3, "max_tokens": 500}
                }
            )
            resp.raise_for_status()
            text = resp.json()["output"]["text"]
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end] if start != -1 else '{"error":"解析失败"}')
    except Exception as e:
        raise HTTPException(status_code=500, detail="分析失败")
