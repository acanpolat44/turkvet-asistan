from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
import pandas as pd
import io
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

app = FastAPI()

# Geçici veri saklama (Railway gibi sunucularda işlem bittikten sonra silinir)
global_df = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload_excel")
async def upload_excel(file: UploadFile = File(...)):
    global global_df
    contents = await file.read()
    global_df = pd.read_excel(io.BytesIO(contents))
    global_df.columns = global_df.columns.str.strip()
    return {"status": "ok"}

@app.post("/ask_question")
async def ask_question(request: QuestionRequest, 
                       x_api_key: str = Header(...), 
                       x_model_name: str = Header(default="gemini-1.5-flash")):
    global global_df
    
    if global_df is None:
        raise HTTPException(status_code=400, detail="Önce bir Excel dosyası yüklenmelidir.")
    
    llm = ChatGoogleGenerativeAI(model=x_model_name, google_api_key=x_api_key, temperature=0)

    agent = create_pandas_dataframe_agent(
        llm, global_df, verbose=True, agent_type="zero-shot-react-description", allow_dangerous_code=True
    )

    try:
        response = agent.invoke(request.question)
        return {"answer": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))