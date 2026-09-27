import ast
import json
import re


def _strip_code_fence(text: str) -> str:
    if "```" not in text:
        return text
    text = re.sub(r"```[a-zA-Z0-9]*", "", text)
    return text.replace("```", "").strip()


def parse_subgoal_response(response) -> list:
    """Parse a VLM response into a flat list of subgoal predicate strings."""
    if isinstance(response, list) and len(response) > 0:
        response = response[0]
    if response is None:
        return []
    text = str(response).strip()
    text = _strip_code_fence(text)

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                # Nested list — return first branch
                return [str(x).strip().strip("'\"") for x in parsed[0] if str(x).strip()]
            return [str(x).strip().strip("'\"") for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Try ast.literal_eval (handles Python list syntax with single quotes)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                return [str(x).strip().strip("'\"") for x in parsed[0] if str(x).strip()]
            return [str(x).strip().strip("'\"") for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Extract list content between [ ]
    if "[" in text and "]" in text:
        content = text[text.find("[") + 1 : text.rfind("]")]
    else:
        content = text

    items = []
    for line in content.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        line = re.sub(r"^\d+[\).]\s*", "", line)
        items.append(line.strip().strip("'\""))
    if not items and "," in content:
        for part in content.split(","):
            part = part.strip()
            if part:
                items.append(part.strip("'\""))
    return items


def parse_branch_response(response) -> list:
    """Parse a VLM response into a list of alternative plan branches.

    Returns a list of lists, where each inner list is one complete plan
    (a sequence of predicate strings). Falls back to a single-branch list
    if the response is not a nested list.
    """
    if isinstance(response, list) and len(response) > 0:
        response = response[0]
    if response is None:
        return []
    text = str(response).strip()
    text = _strip_code_fence(text)

    def _clean_items(lst):
        return [str(x).strip().strip("'\"") for x in lst if str(x).strip()]

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                return [_clean_items(branch) for branch in parsed if branch]
            return [_clean_items(parsed)]
    except Exception:
        pass

    # Try ast.literal_eval
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                return [_clean_items(branch) for branch in parsed if branch]
            # Handle the case where the VLM returns bracket characters as
            # literal string elements: ['[', 'subgoal1', ..., ']', '[', ...]
            if "[" in parsed:
                branches, current = [], []
                for item in parsed:
                    if item == "[":
                        current = []
                    elif item == "]":
                        if current:
                            branches.append(_clean_items(current))
                    else:
                        current.append(item)
                if branches:
                    return branches
            return [_clean_items(parsed)]
    except Exception:
        pass

    # Fall back: treat the whole response as a single plan
    single = parse_subgoal_response(response)
    return [single] if single else []
