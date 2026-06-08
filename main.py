from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.state import ChatRequest, extract_text
from core.graph import app_graph
from core.config import generator_llm, troubleshooting_llm, vision_llm

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "nosana-copilot-python"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    msgs = request.messages
    
    langchain_messages = []
    for m in msgs:
        if m.role == "user":
            if m.image_url:
                langchain_messages.append(HumanMessage(content=[
                    {"type": "text", "text": m.content},
                    {"type": "image_url", "image_url": {"url": m.image_url}}
                ]))
            else:
                langchain_messages.append(HumanMessage(content=m.content))
        else:
            langchain_messages.append(AIMessage(content=m.content))
            
    # Run the LangGraph to collect context and intent
    final_state = await app_graph.ainvoke({"messages": langchain_messages})
    
    async def generate_stream():
        mode = final_state.get("mode")
        needs_clar = final_state.get("needs_clarification")
        context = final_state.get("context", [])
        
        NOSANA_IDENTITY = (
            "You are Nos, an AI assistant from the Nosana team.\n"
            "IMPORTANT FACTS about Nosana you must always follow:\n"
            "- Nosana is a decentralized GPU cloud network where node operators share GPU compute and developers/users run AI inference jobs.\n"
            "- Nosana is NOT a studio. Nosana is NOT a traditional cloud provider. Nosana is a permissionless, decentralized GPU marketplace built on Solana.\n"
            "- Jobs are defined as JSON job definitions and deployed as Docker containers on Nosana nodes.\n"
            "- Users deploy via the Nosana Dashboard (app.nosana.io), the CLI (`nosana` command), or the SDK.\n"
            "- Never describe Nosana as a studio, creative platform, or traditional cloud. Always describe it as a decentralized GPU cloud / compute network.\n"
            "Rules:\n"
            "- Answer questions DIRECTLY. Do NOT ask unnecessary follow-up or clarifying questions for simple factual questions.\n"
            "- Only ask clarifying questions when the user's intent is genuinely ambiguous (e.g. 'help me deploy' without any details).\n"
            "- Keep responses concise. Use markdown formatting with headings where helpful.\n"
            "- For job/deployment templates, always show the JSON job definition structure.\n"
            "- For errors, diagnose the root cause and give a concrete fix.\n"
            "General Guidelines:\n"
            "- Use the provided Context as your primary source of truth for all Nosana-specific features.\n"
            "- If the Context does not contain enough information, use your own broad intelligence to answer the question, but CRITICALLY ALWAYS answer from the perspective of the Nosana ecosystem. Nosana supports a wide variety of NVIDIA consumer and enterprise GPUs (e.g., RTX 4090, 3090, 4080, 3080, A100, H100, RTX A6000, L40S, etc.). Never claim that Nosana only supports 3 GPUs.\n"
            "- If the user asks for help from the team, community, or support, always provide the official Nosana Discord link: https://discord.gg/nosana \n"
            "- Be helpful and try to guide the user towards a solution using both the Context and your internal knowledge, while ensuring the solution involves Nosana.\n"
            "- Think in modern AI and Web3 technical terms. If you encounter short forms or acronyms (like 'MCP'), interpret them in the context of cutting-edge AI (e.g., 'Model Context Protocol') rather than generic cloud terms (like 'Managed Control Plane'), unless the context explicitly says otherwise.\n\n"
        )

        sys_prompt = NOSANA_IDENTITY
        
        if mode == 'greeting':
            sys_prompt += "Greet the user warmly and conversationally like a human. Respond naturally to small talk (e.g., 'Hi! I'm doing well, thanks for asking. How can I help you with Nosana today?'). Do not sound like a rigid robot."
        elif mode == 'clarification' or needs_clar:
            sys_prompt += (
                "Only ask ONE clarifying question to narrow down the user's intent. "
                "Use numbered MCQ options so the user can reply with just a number.\n"
                "Example format:\n"
                "What type of workload are you deploying?\n"
                "1. Custom Docker container\n"
                "2. Pre-built template (PyTorch, ComfyUI, etc.)\n"
                "3. Something else"
            )
        elif mode == 'troubleshooting':
            sys_prompt += f"Diagnose the error and provide a fix.\nFormat: ### Issue\n### Root Cause\n### Fix\n### Next Steps\n\nContext:\n" + "\n".join(context)
        else:
            is_followup = final_state.get("is_followup", False)
            # If RAG returned no useful context, check if it's a follow-up
            if not context or context == ["SYSTEM: Low confidence. Ask user to check Nosana Discord."]:
                if is_followup:
                    sys_prompt += (
                        "No new documentation was found, but this appears to be a follow-up question. "
                        "Answer the user's question directly based on the conversation history above. "
                        "Keep it concise and do not hallucinate new technical details."
                    )
                else:
                    sys_prompt += (
                        "The documentation search returned no relevant results for this exact query. "
                        "Do not immediately decline. Instead, answer the user's question directly using your own internal knowledge. "
                        "CRITICAL: Always tie your answer back to the Nosana Network ecosystem. If asked about GPU recommendations (like for Wan or video generation), "
                        "recommend industry-standard GPUs that are commonly available on the Nosana decentralized compute network (such as RTX 4090, RTX 3090, A100, H100, RTX A6000) "
                        "and mention that they can be rented efficiently on Nosana."
                    )
            else:
                sys_prompt += f"Answer the question directly using the context below if helpful, otherwise rely on your own knowledge. Be concise and use markdown.\n\nContext:\n" + "\n".join(context)
            
        # Check if ANY msg in recent history has an image
        has_image = False
        for msg in langchain_messages[-5:]:
            if isinstance(msg.content, list) and any(isinstance(c, dict) and c.get("type") == "image_url" for c in msg.content):
                has_image = True
                break
        
        if has_image:
            llm_to_use = vision_llm
        elif mode == 'troubleshooting':
            llm_to_use = troubleshooting_llm
        else:
            llm_to_use = generator_llm
            
        # Strip images from history if we are falling back to text-only model
        clean_history = []
        for msg in langchain_messages[-5:]:
            if isinstance(msg.content, list) and not has_image:
                text_only = extract_text(msg.content)
                clean_history.append(HumanMessage(content=text_only) if isinstance(msg, HumanMessage) else AIMessage(content=text_only))
            else:
                clean_history.append(msg)
                
        # Stream
        stream_msgs = [SystemMessage(content=sys_prompt)] + clean_history
        
        try:
            async for chunk in llm_to_use.astream(stream_msgs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            # If vision model fails (bad key, quota, etc), fall back to text model
            print(f"Vision LLM error ({type(e).__name__}), falling back to generator: {e}")
            # Strip image content from messages for text-only fallback
            fallback_msgs = []
            for msg in stream_msgs:
                if isinstance(msg.content, list):
                    text = extract_text(msg.content)
                    fallback_msgs.append(type(msg)(content=text))
                else:
                    fallback_msgs.append(msg)
            async for chunk in generator_llm.astream(fallback_msgs):
                if chunk.content:
                    yield chunk.content
                
    return StreamingResponse(generate_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    # Running on port 8787 so the Vue frontend requires no changes
    uvicorn.run(app, host="0.0.0.0", port=8787)
