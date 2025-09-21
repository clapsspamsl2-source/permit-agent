import os
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("permit-agent")

# Read API key from environment (Render -> Environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment. Set it in Render -> Environment.")
    # We won't raise here so that the app still starts (Render expects an app),
    # but any /ask call will return a clear error if the key is missing.

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Permit Agent", version="1.0.0")

# CORS for testing (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PermitRequest(BaseModel):
    question: str

@app.get("/")
def root():
    """
    Root message so visiting the site doesn't return a raw 404.
    """
    return {
        "message": "Permit Agent service is running. Use /docs for interactive API docs or POST /ask with JSON {\"question\":\"...\"}."
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
async def ask(req: PermitRequest, request: Request):
    """
    Accepts JSON { "question": "..." } and returns {"answer": "..."}.
    """
    if client is None:
        logger.error("OpenAI client not initialized because OPENAI_API_KEY is missing.")
        raise HTTPException(status_code=500, detail="Server misconfigured: OPENAI_API_KEY not set.")

    try:
        # Build system + user messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful permitting assistant. Answer clearly and concisely in plain text. "
                    "If you're unsure, advise checking with the local building department."
                ),
            },
            {"role": "user", "content": req.question},
        ]

        # NOTE: change the model name if needed (e.g. "gpt-4o-mini" or "gpt-4o" depending on access)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )

        # Extract text from response in a safe way
        content = ""
        try:
            content = completion.choices[0].message.content.strip()
        except Exception:
            # fallback: stringify the response for debugging
            logger.exception("Couldn't extract choice text from OpenAI response.")
            content = str(completion)

        return {"answer": content}

    except Exception as e:
        # log full error server-side
        logger.exception("Error while calling OpenAI API")
        # show minimal error to client
        raise HTTPException(status_code=500, detail=f"AI error: {type(e).__name__}: {e}")
