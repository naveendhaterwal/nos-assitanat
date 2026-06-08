import json
from langchain_core.messages import SystemMessage
from core.state import CopilotState, extract_text
from core.config import router_llm, embedding_model, qdrant

async def conversation_analysis_node(state: CopilotState):
    messages = state["messages"]
    last_msg = extract_text(messages[-1].content)
    
    # Get last few messages for context to determine if it's a follow-up
    recent_history = ""
    for m in messages[-4:-1]:
        recent_history += f"{m.type}: {extract_text(m.content)}\n"
        
    prompt = f"""
    Analyze the user's input and determine the conversation mode.
    Modes:
    - greeting: user is just saying hi/hello with no question
    - troubleshooting: user is describing an error, crash, or failure
    - documentation: any question about how Nosana works, features, deployments, jobs, CLI, SDK, API, strategies, etc.
    
    IMPORTANT: Only set needs_clarification=true when the message is completely vague like "help me" or "I need something"
    with NO technical context at all. If the user asks about any specific Nosana concept (even briefly), set needs_clarification=false.
    
    User Input: "{last_msg}"
    
    Respond with JSON ONLY: {{"mode": "str", "needs_clarification": bool}}
    """
    
    try:
        completion = await router_llm.ainvoke([SystemMessage(content=prompt)])
        content = completion.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        parsed = json.loads(content.strip())
        return {
            "mode": parsed.get("mode", "documentation"),
            "needs_clarification": parsed.get("needs_clarification", False)
        }
    except Exception as e:
        print("Analysis error:", e)
        return {"mode": "documentation", "needs_clarification": False}

async def query_refinement_node(state: CopilotState):
    messages = state["messages"]
    last_msg = extract_text(messages[-1].content)
    
    # If the conversation is short, don't overcomplicate
    if len(messages) <= 2:
        return {"search_query": last_msg}
        
    recent_history = ""
    for m in messages[-4:-1]:
        recent_history += f"{m.type}: {extract_text(m.content)}\n"
        
    prompt = f"""
    You are an expert search query refiner for the Nosana Network documentation.
    Given the chat history and the user's latest message, rewrite the user's message into a highly specific, standalone search query for a vector database.
    
    Rules:
    1. If the user's message contains pronouns or references to previous messages (e.g., "what is the difference", "give me an example of that"), replace them with the explicit technical concepts.
    2. Ensure terms like 'Nosana', 'deployment', 'job', etc., are included if relevant.
    3. If the user's message is already specific and standalone, do not change it significantly.
    4. Keep it concise. Do not answer the question. Only output the refined search query.
    
    Recent Chat History:
    {recent_history}
    
    User's Latest Message: "{last_msg}"
    
    Respond with the refined search query ONLY. No preamble, no quotes, no JSON.
    """
    
    try:
        completion = await router_llm.ainvoke([SystemMessage(content=prompt)])
        refined_query = completion.content.strip().strip('"').strip("'")
        print(f"Original Query: {last_msg}")
        print(f"Refined Query: {refined_query}")
        return {"search_query": refined_query}
    except Exception as e:
        print("Refinement error:", e)
        return {"search_query": last_msg}

async def intent_classification_node(state: CopilotState):
    last_msg = extract_text(state["messages"][-1].content)
    prompt = f"""
    Classify the user's request into EXACTLY ONE Qdrant collection:
    - nosana_docs
    - nosana_templates
    - nosana_errors
    - nosana_market
    
    User input: "{last_msg}"
    Respond ONLY with the collection name.
    """
    
    try:
        completion = await router_llm.ainvoke([SystemMessage(content=prompt)])
        return {"intent": completion.content.strip()}
    except:
        return {"intent": "nosana_docs"}

async def clarification_node(state: CopilotState):
    return {"mode": "clarification"}

async def rag_node(state: CopilotState):
    search_query = state.get("search_query")
    if not search_query:
        search_query = extract_text(state["messages"][-1].content)
        
    print(f"RAG Vector Query: {search_query}")
    vector = list(embedding_model.embed([search_query]))[0].tolist()
    
    SCORE_THRESHOLD = 0.35
    collections_to_try = ["nosana_docs"]
    classified = state.get("intent", "nosana_docs")
    if classified and classified != "nosana_docs":
        collections_to_try.append(classified)
    
    best_points = []
    best_score = 0

    for collection in collections_to_try:
        try:
            results = await qdrant.query_points(
                collection_name=collection,
                query=vector,
                limit=5,
                with_payload=True
            )
            points = getattr(results, "points", results)
            if points and points[0].score > best_score:
                best_score = points[0].score
                best_points = points
        except Exception as e:
            print(f"Qdrant error ({collection}):", e)
            continue

    if not best_points or best_score < SCORE_THRESHOLD:
        print(f"RAG low confidence: best_score={best_score:.3f}")
        # Mark as follow-up so the LLM can still answer from conversation history
        return {"context": [], "is_followup": True}
        
    print(f"RAG hit: score={best_score:.3f}")
    return {"context": [r.payload.get("text", "") for r in best_points], "is_followup": False}

async def troubleshooting_node(state: CopilotState):
    last_msg = extract_text(state["messages"][-1].content)
    vector = list(embedding_model.embed([last_msg]))[0].tolist()
    
    try:
        results = await qdrant.query_points(
            collection_name="nosana_errors",
            query=vector,
            limit=5,
            with_payload=True
        )
        points = getattr(results, "points", results)
        return {"context": [r.payload.get("text", "") for r in points], "intent": "nosana_errors"}
    except:
        return {"context": []}

import os
import httpx

async def market_node(state: CopilotState):
    # Hardcode the known public endpoint so it doesn't break if the user puts an API key in the URL env var
    api_url = os.getenv("MARKETS_API_ENDPOINT", "https://dashboard.k8s.prd.nos.ci/api/markets/")
    headers = {}
    
    # If the user added an API key, we can pass it, though the endpoint is currently public
    api_key = os.getenv("MARKETS_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(api_url, headers=headers, timeout=10.0)
            data = resp.json()
            
        market_info = ["### Live Nosana GPU Markets:"]
        for m in data:
            name = m.get("name", "Unknown GPU")
            price = m.get("usd_reward_per_hour", "N/A")
            market_type = m.get("type", "UNKNOWN")
            market_info.append(f"- **{name}** ({market_type}): ${price}/hr")
            
        context_str = "\n".join(market_info)
        return {"context": [context_str], "intent": "nosana_market"}
    except Exception as e:
        print("Market API error:", e)
        return {"context": ["Failed to fetch live market data from API."]}
