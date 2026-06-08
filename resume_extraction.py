import json
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import generator_llm
import os

async def main():
    transcript_file = "discord_transcript_phi_images.json"
    qa_file = "extracted_qa.json"
    
    with open(transcript_file, "r") as f:
        transcript = json.load(f)
        
    print(f"Loaded {len(transcript)} messages with embedded Phi-3.5-vision image texts.")
    
    # Load existing QA pairs to append to
    if os.path.exists(qa_file):
        with open(qa_file, "r") as f:
            qa_pairs = json.load(f)
    else:
        qa_pairs = []
        
    print(f"Loaded {len(qa_pairs)} existing Q&A pairs.")
    
    print("Resuming Q&A Extraction from Chunk 29...")
    total_chunks = len(transcript) // 50 + 1
    
    # Chunk 29 starts at message index 28 * 50 = 1400
    start_index = 28 * 50
    
    for i in range(start_index, len(transcript), 50):
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
        
        IMPORTANT RULE:
        - DO NOT extract any Q&A pairs where a user asks for free SOL or NOS tokens, testnet tokens, or airdrops. Remove these completely.
        
        Return a JSON array of objects. Each object should have:
        - "question": The user's query/issue (include error logs if present).
        - "solution": The team member's answer/fix.
        - "team_member": Who provided the solution.
        
        If there are no valid Q&A pairs in this chunk, return [].
        ONLY RETURN JSON. No markdown ticks, just raw JSON.
        """
        
        try:
            print(f"Processing chunk {chunk_idx} / {total_chunks} [{progress_pct:.1f}%]")
            # Robust backoff for connections
            for attempt in range(5):
                try:
                    res = await generator_llm.ainvoke([
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=chunk_text)
                    ])
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "connection" in err_str or "timeout" in err_str:
                        print(f"API Error (Attempt {attempt+1}/5): {e}. Waiting 60s...")
                        await asyncio.sleep(60)
                    else:
                        raise e
                        
            content = res.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed = json.loads(content)
            if isinstance(parsed, list):
                qa_pairs.extend(parsed)
                
            # Incremental save so we don't lose progress!
            with open(qa_file, "w") as f:
                json.dump(qa_pairs, f, indent=2)
                
        except Exception as e:
            print(f"Error on chunk {i}: {e}")
            
    print(f"Extraction fully complete. Total {len(qa_pairs)} Q&A pairs extracted!")

if __name__ == "__main__":
    asyncio.run(main())
