import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from typing import List, Dict, Any
import time

load_dotenv()

app = FastAPI()

class TextToEmbed(BaseModel):
    input: str
    model: str="default"
    dimensions: int=1445
    encoding_format: str="float"
    user: str="default"


class EmbeddingData(BaseModel):
    # object: str
    embedding: List[float]
    # index: int

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class OpenAIEmbeddingResponse(BaseModel):
    #object: str
    data: List[EmbeddingData]
    #model: str
    #usage: Usage

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/embeddings", response_model=OpenAIEmbeddingResponse)
async def get_embedding(payload: TextToEmbed):
    """
    Accepts a text string and returns an embedding from a local model,
    formatted to match the OpenAI API response.
    """
    print(f"the payload:{payload}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://host.docker.internal:8082/embed",
                json={"inputs": payload.input},
                timeout=30.0,
            )
            response.raise_for_status()
            print(response)
            embedding = response.json()
            print(f"the type: {type(embedding[0][0])}")

        # Note: The token calculation here is a simple approximation.
        # For a more accurate count, a proper tokenizer (e.g., tiktoken) would be needed.
        prompt_tokens = len(payload.input.split())
        total_tokens = 100

        return {"data": [
                    {
                "embedding":embedding[0]
                }
                ]}
        # return {
        #     "object": "list",
        #     "data": [
        #         {
        #             "object": "embedding",
        #             "embedding": embedding[0],
        #             "index": 0
        #         }
        #     ],
        #     "model": "local-embedding-model",
        #     "usage": {
        #         "prompt_tokens": prompt_tokens,
        #         "total_tokens": total_tokens
        #     }
        # }
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to local embedding model: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))