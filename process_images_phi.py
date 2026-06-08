import json
import asyncio
import base64
import httpx
import os

PHI_URL = "https://3wvxtwjtfdywpo7spgcaf4vdqnj3vpplrfa4beavmtfx.node.k8s.prd.nos.ci/v1/chat/completions"

async def process_image(img_url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(img_url, timeout=10.0)
            if resp.status_code != 200:
                print(f"Failed to download {img_url}")
                return ""
            
            b64_image = base64.b64encode(resp.content).decode("utf-8")
            mime_type = resp.headers.get("content-type", "image/png")
            
            prompt = "Extract all text accurately from this screenshot. If it shows an error, log, or terminal output, transcribe it exactly. If it shows a conversation or a solution provided to a problem, capture the context and the solution."
            
            payload = {
                "model": "phi-3.5-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.1
            }
            
            res = await client.post(PHI_URL, json=payload, timeout=60.0)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"Phi API Error {res.status_code}: {res.text}")
                return ""
                
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return ""

async def main():
    transcript_file = "discord_transcript.json"
    output_file = "discord_transcript_phi_images.json"
    
    with open(transcript_file, "r") as f:
        transcript = json.load(f)
        
    print(f"Loaded {len(transcript)} messages. Looking for images...")
    
    processed_count = 0
    total_images = sum(1 for msg in transcript if msg.get("images"))
    
    # Process concurrently with a semaphore to not overload the endpoint (vLLM can handle concurrent requests well)
    sem = asyncio.Semaphore(5)
    
    async def process_msg(msg):
        if not msg.get("images"):
            return
            
        async with sem:
            nonlocal processed_count
            extracted_texts = []
            for img_url in msg["images"]:
                print(f"Processing image {processed_count+1}/{total_images}: {img_url[:50]}...")
                ext = await process_image(img_url)
                if ext:
                    extracted_texts.append(f"[Image Content: {ext}]")
            
            if extracted_texts:
                msg["text"] += "\n\n" + "\n".join(extracted_texts)
            processed_count += 1
            
    # Create tasks for all messages
    tasks = [process_msg(msg) for msg in transcript]
    await asyncio.gather(*tasks)
    
    with open(output_file, "w") as f:
        json.dump(transcript, f, indent=2)
        
    print(f"Successfully processed images and saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
