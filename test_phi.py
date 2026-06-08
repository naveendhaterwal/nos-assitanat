import asyncio
import httpx
import base64

async def test():
    # Use a generic tiny image base64
    b64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    url = "https://3wvxtwjtfdywpo7spgcaf4vdqnj3vpplrfa4beavmtfx.node.k8s.prd.nos.ci/v1/chat/completions"
    payload = {
        "model": "phi-3.5-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this image? Reply very shortly."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }
        ],
        "max_tokens": 50
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=30.0)
        print(resp.status_code, resp.text)

asyncio.run(test())
