"""Explicit planner stop notices, independent of simulator task success."""

import re


def action_limit_notice(limit, count, source):
    return {"reason": "max_actions", "message": f"Reached max actions ({limit}).",
            "limit": limit, "count": count, "limit_source": source}


def cost_cap_notice(cap, spent):
    return {"reason": "llm_cost_cap", "message": f"Reached LLM cost cap (${cap:.2f}).",
            "limit_usd": cap, "spent_usd": round(spent, 6)}


def termination_from_trace(text):
    match = re.search(r"^Termination: (Reached max actions \((\d+)\)\.)\s*$", text, re.M)
    if match:
        return {"reason": "max_actions", "message": match[1], "limit": int(match[2])}
    return {}
