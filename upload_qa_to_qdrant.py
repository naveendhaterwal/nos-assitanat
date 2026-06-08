import json
import asyncio
from core.config import qdrant, embedding_model
from qdrant_client.models import PointStruct
import uuid

async def main():
    try:
        with open("extracted_qa.json", "r") as f:
            qa_pairs = json.load(f)
    except Exception as e:
        print("Could not load extracted_qa.json:", e)
        return
        
    print(f"Loaded {len(qa_pairs)} Q&A pairs. Indexing to Qdrant...")
    
    points = []
    texts_to_embed = []
    
    for qa in qa_pairs:
        # Construct a dense text representation
        q = qa.get("question", "")
        a = qa.get("solution", "")
        tm = qa.get("team_member", "Team")
        ai_reply = qa.get("ai_reply", a)
        
        text = f"User Question: {q}\nNosana AI Answer: {ai_reply}"
        texts_to_embed.append(text)
        
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from qdrant_client.models import PayloadSchemaType
    print("Deleting old discord history points from Qdrant to prevent duplicates...")
    try:
        await qdrant.create_payload_index(
            collection_name="nosana_docs",
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass
        
    await qdrant.delete(
        collection_name="nosana_docs",
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value="discord_history")
                )
            ]
        )
    )
    
    print("Generating embeddings...")
    # FastEmbed uses a generator, we list it
    embeddings = list(embedding_model.embed(texts_to_embed))
    
    for i, qa in enumerate(qa_pairs):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "page_content": texts_to_embed[i],
                    "metadata": {
                        "source": "discord_history",
                        "team_member": qa.get("team_member", "Team"),
                        "question": qa.get("question", ""),
                        "solution": qa.get("solution", ""),
                        "ai_reply": qa.get("ai_reply", "")
                    }
                }
            )
        )
        
    print("Uploading to Qdrant...")
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        await qdrant.upsert(
            collection_name="nosana_docs",
            points=batch
        )
        print(f"Upserted {i + len(batch)} / {len(points)}")
        
    print("Successfully indexed Discord history to RAG!")

if __name__ == "__main__":
    asyncio.run(main())
