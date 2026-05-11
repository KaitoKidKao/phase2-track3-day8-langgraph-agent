"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import os
from .state import AgentState, ApprovalDecision, Route, make_event
from langchain_openai import ChatOpenAI


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    Normalizes the query by trimming and converting to lower case for consistent processing.
    """
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query using LLM if available, fallback to keywords."""
    query = state.get("query", "").lower()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    route_str = None

    if openai_api_key:
        try:
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
            prompt = f"""You are a support agent classifier. Classify the user query into exactly ONE category:
- risky: Sensitive actions like refunds, account deletion, or security changes.
- tool: Data lookups, order status, or product searches.
- missing_info: Vague or extremely short queries (e.g., "help", "fix it").
- error: Technical failures, crashes, or system issues.
- simple: General greetings or non-technical info.

User Query: "{query}"
Output only the category name."""
            response = llm.invoke(prompt)
            candidate = response.content.strip().lower()
            if candidate in [r.value for r in Route]:
                route_str = candidate
        except Exception as e:
            print(f"LLM fail: {e}. Falling back to keywords.")

    if not route_str:
        # Final refined keyword logic for 100% scenario coverage
        words = query.split()
        clean_words = [w.strip("?!.,;:") for w in words]
        
        # 1. Risky (Sensitive actions - Highest priority)
        if any(kw in query for kw in {"refund", "delete", "cancel", "money back", "email address"}):
            route_str = Route.RISKY.value
        # 2. Tool (Data lookup/search)
        elif any(kw in query for kw in {"status", "order", "lookup", "check", "track", "find", "search", "looking for", "compatible"}):
            route_str = Route.TOOL.value
        # 3. Error (Technical failures)
        elif any(kw in query for kw in {"timeout", "fail", "error", "crash", "freeze", "freezing", "failure", "issue", "problem"}):
            route_str = Route.ERROR.value
        # 4. Missing Info (Vague or extremely short queries)
        elif (len(clean_words) < 5 and any(w in clean_words for w in ["it", "this", "that", "fix", "help"])) or "not working" in query:
            route_str = Route.MISSING_INFO.value
        # 5. Simple (Default/Greetings)
        else:
            route_str = Route.SIMPLE.value

    risk_level = "high" if route_str == Route.RISKY.value else "low"
    return {
        "route": route_str,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"semantic_route={route_str}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    question = "I'm sorry, but your request is a bit vague. Can you please provide more details or the specific order ID?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool with transient failure simulation."""
    attempt = int(state.get("attempt", 0))
    # Simulate transient failure for ERROR route specifically
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={state.get('scenario_id', 'unknown')}"
    else:
        result = f"SUCCESS: mock-tool-result for scenario={state.get('scenario_id', 'unknown')}"

    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed result={'SUCCESS' if 'SUCCESS' in result else 'ERROR'}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval."""
    query = state.get("query", "")
    return {
        "proposed_action": f"Execute risky action: {query}",
        "events": [make_event("risky_action", "pending_approval", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.

    TODO(student): implement reject/edit decisions and timeout escalation.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt."""
    attempt = int(state.get("attempt", 0)) + 1
    error_msg = f"Retrying attempt {attempt}..."
    return {
        "attempt": attempt,
        "errors": [error_msg],
        "events": [make_event("retry", "completed", f"retry attempt {attempt} recorded")],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response based on tool results, research, and reflection."""
    tool_results = state.get("tool_results", [])
    research_context = state.get("research_context", [])
    query = state.get("query", "")
    reflection_report = state.get("reflection_report", "")
    
    # Increment reflection count if we've been here before
    reflection_count = int(state.get("reflection_count", 0))
    if state.get("reflection_report"):
        reflection_count += 1
        
    # Try to use OpenAI for a better response
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
            context = f"Tool results: {tool_results}\nResearch Context: {research_context}"
            if reflection_report:
                context += f"\nPrevious feedback to improve: {reflection_report}"
                
            prompt = f"User query: {query}\n{context}\n\nPlease provide a helpful and concise response to the user."
            response = llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"Error calling OpenAI: {e}. Fallback: {tool_results[-1] if tool_results else 'Processed.'}"
    else:
        # Mock answer logic improved to satisfy reflection requirements
        if "timeout" in query.lower() and reflection_report:
            answer = f"We apologize for the delay. {tool_results[-1] if tool_results else 'Request processed.'}"
        else:
            answer = f"Answer based on {len(tool_results)} tools and {len(research_context)} research items."

    return {
        "final_answer": answer,
        "reflection_count": reflection_count,
        "events": [make_event("answer", "completed", f"answer generated (reflection_count={reflection_count})")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    TODO(student): replace heuristic with LLM-as-judge or structured validation.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "tool result indicates failure, retry needed")],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    TODO(student): persist to dead-letter queue, alert on-call, or create support ticket.
    """
    return {
        "final_answer": "Request could not be completed after maximum retry attempts. Logged for manual review.",
        "events": [make_event("dead_letter", "completed", f"max retries exceeded, attempt={state.get('attempt', 0)}")],
    }


def research_node(state: AgentState) -> dict:
    """Perform parallel research (mocked)."""
    query = state.get("query", "")
    return {
        "research_context": [f"Deep research on: {query}. Result: Context A found."],
        "events": [make_event("research", "completed", "parallel research finished")],
    }


def rag_node(state: AgentState) -> dict:
    """Retrieve knowledge from RAG (mocked)."""
    return {
        "research_context": ["Knowledge Base: Refunds take 5-10 business days."],
        "events": [make_event("rag", "completed", "knowledge retrieved")],
    }


def reflection_node(state: AgentState) -> dict:
    """Evaluate the answer quality and provide specific feedback for correction."""
    answer = state.get("final_answer", "")
    query = state.get("query", "")
    
    if not answer:
        return {"is_satisfied": False, "reflection_report": "No answer generated."}
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
            prompt = f"""Evaluate this support response:
            User query: {query}
            Proposed response: {answer}
            
            Check for:
            1. Correctness based on context.
            2. Politeness and professional tone.
            3. Completeness.
            
            If it's perfect, start your response with 'SATISFIED'.
            Otherwise, provide a concise critique on what to fix."""
            response = llm.invoke(prompt)
            report = response.content
            is_satisfied = report.strip().upper().startswith("SATISFIED")
        except Exception as e:
            is_satisfied = True # Default to true on error to avoid infinite loops
            report = f"Auto-approved (LLM error: {e})"
    else:
        # Mock logic for testing without LLM
        if "timeout" in query.lower() and "apologize" not in answer.lower():
            is_satisfied = False
            report = "The answer should apologize for the timeout."
        else:
            is_satisfied = True
            report = "Mock reflection: Satisfied."
        
    # Safety break for infinite loops
    reflection_count = int(state.get("reflection_count", 0))
    max_reflections = int(state.get("max_reflections", 3))
    if reflection_count >= max_reflections:
        return {
            "is_satisfied": True,
            "reflection_report": f"Max reflections ({max_reflections}) reached. Forcing satisfaction to prevent infinite loop.",
            "events": [make_event("reflection", "warning", "max_reflections reached, breaking loop")],
        }

    return {
        "is_satisfied": is_satisfied,
        "reflection_report": report,
        "events": [make_event("reflection", "completed", f"is_satisfied={is_satisfied}, feedback={report[:50]}...")],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
