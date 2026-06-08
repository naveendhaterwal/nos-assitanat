import json
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import generator_llm
import os

async def main():
    file_path = "extracted_qa.json"
    
    with open(file_path, "r") as f:
        qa_pairs = json.load(f)
        
    print(f"Loaded {len(qa_pairs)} Q&A pairs. Adding 'ai_reply' to each...")
    
    # Process in batches to be somewhat fast but safe
    sem = asyncio.Semaphore(5)
    
    async def process_pair(idx, pair):
        if "ai_reply" in pair:
            return
            
        sys_prompt = "You are an expert technical support engineer for the Nosana Network team. Your job is to draft a friendly, helpful, and technically accurate reply to the user's question, using ONLY the provided solution context. Do not invent new information. Be polite, professional, and act as a Nosana team member."
        user_prompt = f"User Question: {pair['question']}\n\nSolution Context: {pair['solution']}\n\nDraft the ideal AI reply for this."
        
        async with sem:
            for attempt in range(5):
                try:
                    res = await generator_llm.ainvoke([
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=user_prompt)
                    ])
                    pair["ai_reply"] = res.content.strip()
                    print(f"[{idx+1}/{len(qa_pairs)}] Generated reply successfully.")
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "connection" in err_str or "timeout" in err_str:
                        print(f"[{idx+1}/{len(qa_pairs)}] Rate limit/Connection error. Retrying in 30s...")
                        await asyncio.sleep(30)
                    else:
                        print(f"[{idx+1}/{len(qa_pairs)}] Failed: {e}")
                        pair["ai_reply"] = "Error generating reply."
                        break

    tasks = [process_pair(i, pair) for i, pair in enumerate(qa_pairs)]
    await asyncio.gather(*tasks)
    
    with open(file_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
        
    print("Finished adding 'ai_reply' to all questions!")

if __name__ == "__main__":
    asyncio.run(main())
