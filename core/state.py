from typing import TypedDict, Annotated, Optional, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

def extract_text(content: Any) -> str:
    if isinstance(content, list):
        return next((c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"), "")
    return str(content)

def override_messages(existing, new):
    return new

class CopilotState(TypedDict):
    messages: Annotated[list[BaseMessage], override_messages]
    mode: str
    needs_clarification: bool
    intent: str
    context: list[str]
    is_followup: bool
    search_query: str

class MessageItem(BaseModel):
    role: str
    content: str
    image_url: Optional[str] = None

class ChatRequest(BaseModel):
    messages: list[MessageItem]
