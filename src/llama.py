import httpx, os
from pydantic import BaseModel
from asyncio import sleep


async def pause_llm():
    llama_url = os.getenv("LLAMA_URL")
    if not llama_url:
        print("⚠️ LLAMA_URL not set; skipping LLM pause.")
        return
    print("Attempting to pause LLM inference via API...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{llama_url}/pause", timeout=50)
            if response.status_code == 200:
                print("LLM inference paused successfully.")
            else:
                print(
                    f"Failed to pause LLM inference. Status code: {response.status_code}"
                )

    except Exception as e:
        print(f"Error while trying to pause LLM inference: {e}")
