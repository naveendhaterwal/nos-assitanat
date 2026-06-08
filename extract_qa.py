import json
import asyncio
import base64
import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import generator_llm, vision_llm

TEAM_MEMBERS = {"ersguteralbaner", "denis255", "seanster8290", "djmbritt"}

async def process_image(url: str):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)
            if resp.status_code != 200: return ""
            b64_image = base64.b64encode(resp.content).decode("utf-8")
            mime_type = resp.headers.get("content-type", "image/png")
            
            prompt = "Extract all text accurately from this screenshot. Describe any errors shown."
            
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
            ])
            
            res = await vision_llm.ainvoke([msg])
            return res.content.strip()
    except Exception as e:
        print(f"Failed to process image {url}: {e}")
        return ""

async def main():
    with open("discord_transcript.json", "r") as f:
        transcript = json.load(f)
        
    print(f"Loaded {len(transcript)} messages.")
    
    print("Skipping images due to Gemini rate limits to ensure fast extraction...")
                
    # 2. Extract Q&A using generator LLM
    # We will chunk the transcript by 50 messages.
    qa_pairs = []
    
    total_chunks = len(transcript) // 50 + 1
    
    for i in range(0, len(transcript), 50):
        chunk_idx = i // 50 + 1
        progress_pct = (chunk_idx / total_chunks) * 100
        
        chunk = transcript[i:i+50]
        chunk_text = ""
        for m in chunk:
            chunk_text += f"{m['name']} ({m['time']}): {m['text']}\n"
            
        sys_prompt = f"""
        You are an expert technical documentation assistant. Extract all Q&A pairs from this Discord chat transcript.
        The core team members answering questions are: ersguteralbaner, denis255, seanster8290, djmbritt.
        
        Look for instances where a normal user asks a question/reports an error, and a core team member provides a solution/answer.
        Combine context if there's back-and-forth.
        
        Return a JSON array of objects. Each object should have:
        - "question": The user's query/issue (include error logs if present).
        - "solution": The team member's answer/fix.
        - "team_member": Who provided the solution.
        
        If there are no valid Q&A pairs in this chunk, return [].
        ONLY RETURN JSON. No markdown ticks, just raw JSON.
        """
        
        try:
            print(f"Processing chunk {chunk_idx} / {total_chunks} [{progress_pct:.1f}%]")
            res = await generator_llm.ainvoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=chunk_text)
            ])
            
            content = res.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed = json.loads(content)
            if isinstance(parsed, list):
                qa_pairs.extend(parsed)
        except Exception as e:
            print(f"Error on chunk {i}: {e}")
            
    with open("extracted_qa.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)
        
    print(f"Extracted {len(qa_pairs)} Q&A pairs total.")

if __name__ == "__main__":
    asyncio.run(main())
