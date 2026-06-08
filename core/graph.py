from langgraph.graph import StateGraph, START, END
from core.state import CopilotState
from core.nodes import (
    conversation_analysis_node, 
    intent_classification_node, 
    clarification_node, 
    query_refinement_node,
    rag_node, 
    troubleshooting_node,
    market_node
)

def route_after_analysis(state: CopilotState):
    if state.get("needs_clarification"): return "clarification"
    if state.get("mode") == "greeting": return "END" # skip to streaming
    return "refinement"

def route_after_intent(state: CopilotState):
    intent = state.get("intent", "")
    if "market" in intent: return "market"
    if state.get("mode") == "troubleshooting": return "troubleshooting"
    return "rag"

def build_graph():
    workflow = StateGraph(CopilotState)
    workflow.add_node("analysis", conversation_analysis_node)
    workflow.add_node("refinement", query_refinement_node)
    workflow.add_node("intent", intent_classification_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("troubleshooting", troubleshooting_node)
    workflow.add_node("market", market_node)

    # The graph ends after accumulating context. The actual streaming response
    # is handled by FastAPI immediately after the graph finishes execution.
    workflow.add_edge(START, "analysis")
    workflow.add_conditional_edges("analysis", route_after_analysis, {
        "clarification": "clarification",
        "END": END,
        "refinement": "refinement"
    })
    workflow.add_edge("refinement", "intent")
    workflow.add_edge("clarification", END)
    workflow.add_conditional_edges("intent", route_after_intent, {
        "troubleshooting": "troubleshooting",
        "rag": "rag",
        "market": "market"
    })
    workflow.add_edge("rag", END)
    workflow.add_edge("troubleshooting", END)
    workflow.add_edge("market", END)

    return workflow.compile()

app_graph = build_graph()
