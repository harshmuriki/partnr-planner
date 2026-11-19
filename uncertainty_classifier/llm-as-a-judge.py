import argparse
import csv
import gzip
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from classify_errors import Run

TAXONOMY = {
    "Memory": {
        "Over-simplification / Incomplete Summary": "Summarizes past info too crudely, ignoring details; leads to flawed reasoning.",
        "Hallucination (False Memory)": "Recalls events or states that never happened, filling missing gaps with fabricated info.",
        "Retrieval Failure": "Relevant info exists but is not retrieved when needed.",
    },
    "Reflection": {
        "Progress Misassessment": "Incorrectly evaluates progress (too optimistic, too pessimistic, or misses completion).",
        "Outcome Misinterpretation": "Executes an action but misreads the immediate result or environment feedback.",
        "Causal Misattribution": "Correctly notes failure but blames the wrong cause, misguiding subsequent plans.",
        "Hallucination": "Reflects on events/results that never occurred.",
    },
    "Planning": {
        "Constraint Ignorance": "Ignores limits (time, budget, space, etc.) when forming plans.",
        "Impossible Action": "Plans a step that is physically/logically impossible given current preconditions.",
        "Inefficient Planning": "Plan is overly long or illogical; wastes steps and risks hitting limits.",
        "Goal Misinterpretation": "Misconstruing the goal or removing/changing necessary parts of it.",
    },
    "Action": {
        "Planning–Action Disconnect": "Chosen actions do not align with the stated plan intent.",
        "Format Error": "Produces syntactically invalid actions.",
        "Parameter Error": "Generates unreasonable or malformed parameters.",
    },
    "System": {
        "Step Limit Exhaustion": "Fails due to reaching the maximum step cap despite reasonable behavior.",
        "Tool Execution Error": "External tool/API misbehaves or errors, causing downstream failures.",
        "LLM Limit": "Fails due to API/model constraints (e.g., timeouts, token limits).",
        "Environment Error": "Simulator/environment breaks expected rules (bug/crash/network), not agent’s fault.",
        "Framework Limitation": "The model produced usable output, but the planner's own code could not use it: its goal, hypothesis or PDDL format has no way to express a requirement, its prompt rules forced an unusable answer, or its control logic rejected the output or ended the run by its own rule.",
    },
}

EXAMPLES = {
    "Over-simplification / Incomplete Summary": [
        {
            "context": "tru-pomdp, T6 (heat the bread, bring it with a filled water bottle and a clean hand towel to the bedroom desk), INC memory, DIS variant: a second bread and a dirty towel are distractors.",
            "observed_error": "Wrong Object: bread_25_0 (distractor): Pick, Place; ended on table_11 in bedroom_3",
            "attribution": {
                "module": "Memory",
                "error_type": "Over-simplification / Incomplete Summary",
                "step": "a0",
                "evidence": '"hypothesized": ["bread_25_0", "dirty_hand_towel_0", "lamp_living_0", "water_bottle_0"]',
                "description": "The goal hypotheses collapsed the requested objects into coarse category matches: the distractor bread stood in for the target bread and the dirty towel for the clean one, so the distractor was delivered.",
                "confidence": 0.8,
            },
        },
    ],
    "Hallucination (False Memory)": [
        {
            "context": "react, T4 (put the three fruits in the basket and the two drink containers on the counter), ACC memory, ABS variant.",
            "observed_error": "Precondition Error: a21 Place[apple_1, on, basket_0, None, None]: Unexpected failure! - Failed to place! The agent is not holding any object.",
            "attribution": {
                "module": "Memory",
                "error_type": "Hallucination (False Memory)",
                "step": "a21",
                "evidence": "Place[apple_1, within, basket_0, None, None] -> Successful execution! Then: Thought: The previous “within” placement did not register, so I will try placing the held apple directly on the fruit basket as an alternate valid placement.",
                "description": "After the apple had been placed successfully, the planner still believed it was holding it and issued another place with an empty hand.",
                "confidence": 0.9,
            },
        },
    ],
    "Retrieval Failure": [
        {
            "context": "react, T5 (clean and fill both glasses with the soap next to each glass before cleaning, put them and a plate on the living-room side table), OUT memory, DIS variant.",
            "observed_error": "Missing Prerequisite: is_next_to(soap_dispenser_0, glass_0): never done, needed before is_clean(glass_0)",
            "attribution": {
                "module": "Memory",
                "error_type": "Retrieval Failure",
                "step": "a5",
                "evidence": "soap_dispenser_0: on cabinet_0 in kitchen_1. Thought: I am now near the kitchen cabinet faucet, which is required for cleaning. I will clean the second dirty glass first while at the faucet.",
                "description": "The observation showed the real soap dispenser right where the robot stood, but the planner never used it and cleaned the glass without placing the soap next to it first.",
                "confidence": 0.8,
            },
        },
    ],
    "Progress Misassessment": [
        {
            "context": "react, T3 (place two green books on the living-room side table, turn off the laptop), INC memory, BASE variant.",
            "observed_error": "Incorrect Placement: book_0 ended on table_3 in living_room_1; unmet: is_on_top(book_0, table_0)",
            "attribution": {
                "module": "Reflection",
                "error_type": "Progress Misassessment",
                "step": "a11",
                "evidence": "Thought: Both green books are on the living room side table, and the laptop has been turned off. The task is complete. Done[]",
                "description": "The planner declared the task complete although both books were on table_3, not the required table_0, so it never corrected the placement.",
                "confidence": 0.8,
            },
        },
    ],
    "Outcome Misinterpretation": [
        {
            "context": "react, T1 (turn off the bedroom lights, bring a full jug of water to a living-room table), INC memory, CON variant: the jug is inside the closed fridge.",
            "observed_error": "Object Not Found: jug_0: never seen on/in fridge_0 in kitchen_1 (explored kitchen_1 at a0)",
            "attribution": {
                "module": "Reflection",
                "error_type": "Outcome Misinterpretation",
                "step": "a3",
                "evidence": "Thought: ... The jug was not found in the kitchen, so I will exhaustively search the laundry room ...",
                "description": "The planner read an exploration that did not reveal the jug as proof the jug was not in the kitchen, ignoring that it could be inside the closed fridge, and moved on to other rooms.",
                "confidence": 0.8,
            },
        },
    ],
    "Causal Misattribution": [
        {
            "context": "react, T6 (heat the bread, bring it with a filled water bottle and a clean hand towel to the bedroom desk), OUT memory, CON variant.",
            "observed_error": "Time Limit Reached: Stopped after 1200 s",
            "attribution": {
                "module": "Reflection",
                "error_type": "Causal Misattribution",
                "step": "a11",
                "evidence": "Thought: The bread and bottle are no longer available after the failed pickup attempts. ... Pick[hand_towel_1] -> Unexpected failure! - Failed to pick! Object does not exist.",
                "description": "The planner blamed its failed pickups on objects having been removed from the scene, instead of on its outdated memory, so it kept trusting stale locations and never searched the closed furniture until time ran out.",
                "confidence": 0.7,
            },
        },
    ],
    "Hallucination": [
        {
            "context": "react, T1 (turn off the bedroom lights, bring a full jug of water to a living-room table), ACC memory, ABS variant: the jug is known to be absent.",
            "observed_error": "None. The run's only observed error is Unnecessary Action: a2 Explore[bedroom_1]; the Close at a3 is not flagged by the rules, so this is reported with error_id \"extra\".",
            "attribution": {
                "error_id": "extra",
                "module": "Reflection",
                "error_type": "Hallucination",
                "step": "a3",
                "evidence": "Thought: The bedroom lamp is off, and the bedroom has been exhaustively checked. ... I will close the chest of drawers left open during the search. Close[chest_of_drawers_0]",
                "description": "The planner reflected on a step that never happened: no action opened the chest of drawers, yet it closed it as if it had been left open during the search. The wasted Close is a genuine error that none of the observed errors covers.",
                "confidence": 0.7,
            },
        },
    ],
    "Constraint Ignorance": [
        {
            "context": "Illustrative (not from a real run). react, T5, ACC memory, BASE variant. The robot can hold only one object at a time.",
            "observed_error": "Precondition Error: a6 Pick[glass_1]: Unexpected failure! - Failed to pick! The agent cannot hold more than one object at a time.",
            "attribution": {
                "module": "Planning",
                "error_type": "Constraint Ignorance",
                "step": "a6",
                "evidence": "Thought: I'll carry both glasses to the sink at once to save trips. Pick[glass_1] (while holding glass_0)",
                "description": "The plan ignored the robot's one-object carrying limit and tried to pick up a second glass while already holding one.",
                "confidence": 0.8,
            },
        },
    ],
    "Impossible Action": [
        {
            "context": "react, T1 (turn off the bedroom lights, bring a full jug of water to a living-room table), INC memory, CON variant.",
            "observed_error": "Affordance Error: a12 Open[cabinet_0]: Unexpected failure! - Failed to Open! cabinet_0 is not articulated - and cannot be Opened.",
            "attribution": {
                "module": "Planning",
                "error_type": "Impossible Action",
                "step": "a12",
                "evidence": "Thought: I am in the kitchen, where the jug is most likely stored. I will open cabinet_0 to inspect inside for a jug, then close it afterward if needed. Open[cabinet_0]",
                "description": "The planner planned to open a cabinet that cannot be opened, a step that is physically impossible in this scene.",
                "confidence": 0.8,
            },
        },
    ],
    "Inefficient Planning": [
        {
            "context": "react, T4 (put the three fruits in the basket and the two drink containers on the counter), INC memory, CON variant: the items are inside the closed fridge.",
            "observed_error": "Time Limit Reached: Stopped after 600 s",
            "attribution": {
                "module": "Planning",
                "error_type": "Inefficient Planning",
                "step": "a1",
                "evidence": "Thought: The kitchen and living room contain no fruits or drink containers, so I should search the lounge next for the missing objects. ... Thought: ... I should continue the exhaustive search in bathroom_2.",
                "description": "Instead of opening the kitchen fridge, the planner searched unrelated rooms one by one and used up the time limit.",
                "confidence": 0.7,
            },
        },
    ],
    "Goal Misinterpretation": [
        {
            "context": "vlm-tamp, T3 (place two green books on the living-room side table, turn off the laptop), ACC memory, ABS variant: book_1 is known to be absent.",
            "observed_error": "Task Abandoned: unsolvable; never attempted: is_on_top(book_0, table_0), is_powered_off(laptop_0)",
            "attribution": {
                "module": "Planning",
                "error_type": "Goal Misinterpretation",
                "step": "start",
                "evidence": "Planner decision: unsolvable. Only book_0 is available; book_1 is explicitly known absent ... collecting two books is impossible, although laptop_0 can be turned off and book_0 can be placed on table_0.",
                "description": "The planner treated one absent book as making the whole task impossible and dropped the parts it said itself were achievable.",
                "confidence": 0.9,
            },
        },
        {
            "context": "vlm-tamp, T3 (same task), INC memory, BASE variant.",
            "observed_error": "Incorrect Placement: book_0 ended on table_3 in living_room_1; unmet: is_on_top(book_0, table_0)",
            "attribution": {
                "module": "Planning",
                "error_type": "Goal Misinterpretation",
                "step": "a6",
                "evidence": "VLM English plan: ... Carry book_0 to table_3 in living_room_1. Place book_0 on table_3. ... Carry book_1 to table_3 in living_room_1. Place book_1 on table_3.",
                "description": "The plan consistently grounded 'the living-room side table' to table_3 instead of the required table_0. The action matched the plan, so this is a wrong reading of the goal, not a Planning–Action Disconnect.",
                "confidence": 0.8,
            },
        },
    ],
    "Planning–Action Disconnect": [
        {
            "context": "Illustrative (not from a real run). react, T5, ACC memory, BASE variant.",
            "observed_error": "Incorrect Placement: glass_0 ended on table_3 in living_room_1; unmet: is_on_top(glass_0, table_0)",
            "attribution": {
                "module": "Action",
                "error_type": "Planning–Action Disconnect",
                "step": "a8",
                "evidence": "Thought: Glass_0 is clean and filled. I will carry it to the living-room side table, table_0, and place it there. Place[glass_0, on, table_3, None, None]",
                "description": "The stated intent named the correct table_0, but the action placed the glass on table_3. Had the thought also said table_3, it would be Goal Misinterpretation.",
                "confidence": 0.9,
            },
        },
    ],
    "Format Error": [
        {
            "context": "Illustrative (not from a real run). react, T5, ACC memory, BASE variant.",
            "observed_error": "Time Limit Reached: Stopped after 600 s",
            "attribution": {
                "module": "Action",
                "error_type": "Format Error",
                "step": "a10",
                "evidence": "Thought: I'll place the glass on the side table. Place[glass_0 on table_0] (repeated until the time limit)",
                "description": "The action was written without the required commas, so it could not be parsed; the planner kept repeating the malformed action until time ran out.",
                "confidence": 0.8,
            },
        },
    ],
    "Parameter Error": [
        {
            "context": "react, T4 (put the three fruits in the basket and the two drink containers on the counter), ACC memory, BASE variant.",
            "observed_error": "Affordance Error: a7 Place[apple_0, on, basket_0, None, None]: Unexpected failure! - Failed to place! Destination must be furniture, floor, or an annotated container used with within.",
            "attribution": {
                "module": "Action",
                "error_type": "Parameter Error",
                "step": "a7",
                "evidence": "Thought: The basket placement has not registered with the “within” relation. I will place the held apple on the basket as a surface instead. Place[apple_0, on, basket_0, None, None]",
                "description": "The relation parameter 'on' is invalid for a basket, which only accepts 'within', so the place could never succeed.",
                "confidence": 0.8,
            },
        },
    ],
    "Step Limit Exhaustion": [
        {
            "context": "react, T4 (put the three fruits in the basket and the two drink containers on the counter), OUT memory, CON variant.",
            "observed_error": "Time Limit Reached: Stopped after 601 s",
            "attribution": {
                "module": "System",
                "error_type": "Step Limit Exhaustion",
                "step": "a31",
                "evidence": "a29 Place[bottle_0, on, counter_0, None, None]: Successful execution! a30 Navigate[fridge_0]: Successful execution! a31 Pick[bottle_1]: Successful execution!",
                "description": "Every recent action succeeded and the task was 83% complete; the robot was delivering the last bottle when the 600 s limit ended the run.",
                "confidence": 0.8,
            },
        },
    ],
    "Tool Execution Error": [
        {
            "context": "tru-pomdp, T7 (put away the toys, wash and put away the dirty dish, turn off the lights), INC memory, BASE variant.",
            "observed_error": "Time Limit Reached: Stopped after 1201 s",
            "attribution": {
                "module": "System",
                "error_type": "Tool Execution Error",
                "step": "a2",
                "evidence": "Action: Explore[living_room_1] Result: Unexpected failure! - Skill took too long to finish.",
                "description": "The Explore skill timed out without exploring, so the planner could not discover the objects it needed and the run stalled until the time limit.",
                "confidence": 0.7,
            },
        },
    ],
    "LLM Limit": [
        {
            "context": "vlm-tamp, T6 (heat the bread, bring it with a filled water bottle and a clean hand towel to the bedroom desk, turn off the living-room light), ACC memory, DIS variant.",
            "observed_error": "Missing Action: is_powered_off(lamp_living_0): never done",
            "attribution": {
                "module": "System",
                "error_type": "LLM Limit",
                "step": "start",
                "evidence": "Planner decision: invalid_response. Empty VLM plan; inspect completion token limit",
                "description": "The model returned an empty plan because it hit its completion-token limit, so the robot never acted.",
                "confidence": 0.9,
            },
        },
    ],
    "Environment Error": [
        {
            "context": "Illustrative (not from a real run). react, T2 (put both packages next to each other on the living-room table with the scissors beside them), ACC memory, BASE variant.",
            "observed_error": "Incorrect Placement: box_0 ended on floor_living_room_1 in living_room_1; unmet: is_on_top(box_0, table_2)",
            "attribution": {
                "module": "System",
                "error_type": "Environment Error",
                "step": "a4",
                "evidence": "Place[box_0, on, table_2, None, None] -> Successful execution! Next observation: box_0: on floor_living_room_1, with no robot action in between.",
                "description": "The place was valid and reported success, but the box then fell through the table surface by itself; the simulator, not the robot, broke the expected behaviour.",
                "confidence": 0.8,
            },
        },
    ],
    "Framework Limitation": [
        {
            "context": "tru-pomdp, T1 (turn off the bedroom lights, bring a full jug of water to a living-room table), ACC memory, ABS variant: the jug is known to be absent.",
            "observed_error": "Task Abandoned: empty_belief; never attempted: is_powered_off(lamp_0)",
            "attribution": {
                "module": "System",
                "error_type": "Framework Limitation",
                "step": "start",
                "evidence": "Tree of Hypotheses returned no hypotheses; stopping.",
                "description": "The hypothesis prompt requires every object in the instruction ('never truncate a combination'), so with the jug known to be absent the model correctly returned no hypotheses and the planner stopped before turning off the lamp.",
                "confidence": 0.8,
            },
        },
    ],
}
INSTRUCTIONS = """You are an expert at analyzing why household-robot planner runs fail or waste effort.

You are given one run: the task, the variant being tested, the robot's starting memory, how the run ended,
every action it took (a1..aN), the planner's trace, and a list of observed errors (e1..eN) that rule-based
checks already found in it.

Your job: for EACH observed error, identify the underlying cause(s) using the taxonomy below.

Rules
1. Use the taxonomy carefully. Give the specific error type, not just the module, and only an error type
   listed under that module.
2. Every observed error must get at least one attribution. If several causes contributed, give one
   attribution per cause with the same error_id.
   In special cases you may also report a genuine error at a step that none of the observed errors covers
   (for example a wasted action the rules did not flag), with error_id "extra". Use this sparingly: only for
   clear errors with direct evidence in the run, not to re-check every step.
3. For "step", give the action id where the cause FIRST shows, not where its effect appears. If the cause
   shows before the first action (for example the planner stopped before acting), use "start".
4. Think about the whole run before deciding: trace each observed error back to the decision that led to it.
5. Read the planner's trace, not just the actions. It shows the model's reasoning and the planner's
   internals (for example ReAct's thoughts, VLM-TAMP's subgoals and PDDL plans, TRU-POMDP's goal
   hypotheses and beliefs), which often reveal causes the actions alone do not, such as a goal translated
   wrongly into PDDL or a wrong hypothesis about where an object belongs.
6. Base every attribution on what the run shows, and support it with evidence from the actions or the trace.
7. If the run does not show a cause, use the "no cause" output instead of guessing.
8. System errors are valid causes when the robot behaved reasonably and the framework, tools or limits
   were at fault.

About these runs
9. The variant's uncertainty is deliberate. Outdated (OUT) or incomplete (INC) memory, absent (ABS) or
   hidden (CON) objects, distractors (DIS) and substitutes (SUB) are the test, not errors in themselves.
   The simulator behaving as the variant describes is not an Environment Error.
10. Acting on the starting memory is not Hallucination (False Memory), even if that memory is outdated.
    Failing to update after the simulator contradicts it can be an error.
11. The observed errors are fixed. Do not relabel or dispute them; explain their causes.

Framework Limitation vs other types
12. Use Framework Limitation only when the model's output was usable (or was forced by the planner's own
    rules) and the planner itself could not express or act on it. Ask: would the same model output have
    worked in a planner that could express it?
13. If the model's own output contradicts its own stated plan, use Planning–Action Disconnect.
14. If the model asked for something physically or logically impossible, use Impossible Action.
15. If the model dropped or changed part of the goal by its own choice, use Goal Misinterpretation.
16. If a step cap or time limit ended a reasonable run, use Step Limit Exhaustion.
17. If a skill or API call failed or misbehaved, use Tool Execution Error.
18. If the simulator broke its own rules, use Environment Error.
19. For Framework Limitation, the evidence must point to the planner's own behaviour in the trace (for
    example a "Planner decision:" line, a PDDL goal missing a requirement, or "Tree of Hypotheses returned
    no hypotheses").

Output
20. Return only JSON in the template below. No markdown, no other text."""
OUTPUT_TEMPLATE = """{
    "reasoning": "[STEP-BY-STEP ANALYSIS OF THE RUN BEFORE LABELLING]",
    "attributions": [
        {
            "error_id": "[OBSERVED ERROR BEING EXPLAINED, e.g. e1, or extra for a genuine error none of them covers]",
            "module": "[Memory | Reflection | Planning | Action | System]",
            "error_type": "[AN ERROR TYPE FROM THE TAXONOMY UNDER THAT MODULE]",
            "step": "[ACTION WHERE THE CAUSE FIRST SHOWS, e.g. a12, or start if before the first action]",
            "evidence": "[QUOTE OR OBSERVATION FROM THE RUN SUPPORTING THIS]",
            "description": "[WHY THIS IS THE CAUSE OF THE OBSERVED ERROR]",
            "confidence": [0.0 TO 1.0]
        },
        ... # one or more per observed error; repeat the error_id for several causes
    ]
}"""
EXAMPLE_OUTPUT = """Run: react, T2 ("Move both packages to the living-room center table and place them next to each other.
Bring the scissors and place beside them to open the packages."), ACC memory, DIS variant. The required table is table_2.
Observed errors:
e1 failure / Action Undone: is_on_top(box_0, table_2): met at step 500, not true at the end
e2 failure / Incorrect Placement: box_1 ended on table_1 in living_room_1; unmet: is_on_top(box_1, table_2)
e3 failure / Incorrect Placement: scissors_0 ended on table_1 in living_room_1; unmet: is_on_top(scissors_0, table_2)
e4 inefficiency / Unnecessary Action: a1 Explore[living_room_1]: Successful execution!

Answer:
{
    "reasoning": "The task needs both packages and the scissors on the living-room center table (table_2). At a2 the planner decided 'The living-room center table appears to be table_1' and then placed every object on table_1, so e2 and e3 come from that choice. e1 has the same cause: box_0 only counted as on table_2 at step 500, while it was being carried during the a2 navigation; it was then placed on table_1 at a3 and never moved again. e4: with accurate memory the furniture was already known, so exploring the living room at a1 only to identify the table was unnecessary.",
    "attributions": [
        {
            "error_id": "e1",
            "module": "Planning",
            "error_type": "Goal Misinterpretation",
            "step": "a2",
            "evidence": "Thought: The living-room center table appears to be table_1, so I will navigate to it while carrying box_0.",
            "description": "The planner grounded 'the living-room center table' to table_1 instead of table_2. box_0 only briefly counted as on table_2 while it was carried past, and it ended on table_1.",
            "confidence": 0.7
        },
        {
            "error_id": "e2",
            "module": "Planning",
            "error_type": "Goal Misinterpretation",
            "step": "a2",
            "evidence": "Thought: The living-room center table appears to be table_1 ... Place[box_1, on, table_1, next_to, box_0]",
            "description": "The same wrong table choice made at a2 sent box_1 to table_1.",
            "confidence": 0.85
        },
        {
            "error_id": "e3",
            "module": "Planning",
            "error_type": "Goal Misinterpretation",
            "step": "a2",
            "evidence": "Thought: The living-room center table appears to be table_1 ... Place[scissors_0, on, table_1, next_to, box_0]",
            "description": "The same wrong table choice sent the scissors to table_1.",
            "confidence": 0.85
        },
        {
            "error_id": "e4",
            "module": "Planning",
            "error_type": "Inefficient Planning",
            "step": "a1",
            "evidence": "Thought: I am holding box_0 and need to identify the living-room center table before placing it, so I will explore the living room while carrying the package.",
            "description": "With accurate memory the living-room furniture was already known, so the exploration added a step without new information.",
            "confidence": 0.7
        }
    ]
}"""
NO_CAUSE_OUTPUT = """If the run does not show a cause for an observed error, still give it one attribution, like this:
{
    "error_id": "[THE OBSERVED ERROR, e.g. e3]",
    "module": null,
    "error_type": null,
    "step": null,
    "evidence": "[WHAT YOU CHECKED IN THE ACTIONS AND TRACE]",
    "description": "[WHY NO CAUSE CAN BE IDENTIFIED FROM THE RUN]",
    "confidence": [0.0 TO 1.0]
}"""

RETRY_WARNING = """IMPORTANT: Your previous answer was invalid:
{problems}
Fix these and answer again. Every observed error needs at least one attribution, every error type must be
listed under its module in the taxonomy, and every step must be "start" or an action id from this run."""
MODEL = "gpt-6-astra"
MAX_RETRIES = 2
MAX_TOKENS = 32000

API_URL = "https://api.openai.com/v1/chat/completions"
KEY_FILE = Path(__file__).resolve().parent / ".openai_api_key"

def load_run(row, findings):
    run = Run(row)
    episode = json.loads(gzip.open(run.folder / "input" / "dataset.json.gz").read())["episodes"][0]
    log = (run.folder / "log.txt").read_text(errors="ignore")
    memory = log.split("Robot Scene Graph:", 1)[1].split("\n\n", 1)[0] if "Robot Scene Graph:" in log else ""
    termination = (row["termination"] or {}).get("message", "")
    return {
        "task": episode["instruction"],
        "variant": f"{row['memory']} memory, {row['uncertainty']} variant\n{(run.folder / 'input' / 'spec.md').read_text()}",
        "memory": re.sub(r"\x1b\[[0-9;]*m", "", memory).strip(),
        "outcome": f"{'succeeded' if row['task_state_success'] == 1 else 'failed'}; ended: {row['execution_status']} {termination}; {row['task_explanation']}",
        "actions": "\n".join(str(action) for action in run.actions),
        "trace": run.trace,
        "errors": [f"e{i} {finding['kind']} / {finding['label']}: {finding['evidence']}" for i, finding in enumerate(findings, 1)],
        "action_ids": {f"a{action.id}" for action in run.actions},
    }

def build_prompt(run_input, problems=None):
    taxonomy = "\n".join(
        f"{module}\n" + "\n".join(f"- {name}: {definition}" for name, definition in types.items())
        for module, types in TAXONOMY.items()
    )
    examples = "\n\n".join(json.dumps(example, indent=1) for examples in EXAMPLES.values() for example in examples)
    sections = [
        INSTRUCTIONS,
        "# Taxonomy\n" + taxonomy,
        "# Worked examples (one per error type)\n" + examples,
        "# Output format\n" + OUTPUT_TEMPLATE,
        "# Example answer for a whole run\n" + EXAMPLE_OUTPUT,
        "# When no cause can be found\n" + NO_CAUSE_OUTPUT,
    ]
    if problems:
        sections.append(RETRY_WARNING.format(problems="\n".join(f"- {problem}" for problem in problems)))
    sections += [
        "# The run",
        "Task: " + run_input["task"],
        "Variant:\n" + run_input["variant"],
        "Robot's starting memory:\n" + run_input["memory"],
        "Outcome: " + run_input["outcome"],
        "Actions:\n" + run_input["actions"],
        "Planner trace:\n" + run_input["trace"],
        "Observed errors:\n" + "\n".join(run_input["errors"]),
    ]
    return "\n\n".join(sections)

def check(answer, run_input):
    if not isinstance(answer, dict) or not isinstance(answer.get("attributions"), list):
        return ["the answer is not a JSON object with an attributions list"]
    error_ids = {f"e{i}" for i in range(1, len(run_input["errors"]) + 1)}
    pairs = {(module, name) for module, types in TAXONOMY.items() for name in types}
    problems = [f"{error_id} has no attribution" for error_id in sorted(error_ids - {a.get("error_id") for a in answer["attributions"]})]
    for attribution in answer["attributions"]:
        error_id = attribution.get("error_id")
        fields = [attribution.get("module"), attribution.get("error_type"), attribution.get("step")]
        if error_id != "extra" and error_id not in error_ids:
            problems.append(f"{error_id}: unknown error_id")
        if all(field is None for field in fields):
            continue
        if any(field is None for field in fields):
            problems.append(f"{error_id}: module, error_type and step must all be filled in, or all be null")
            continue
        if (fields[0], fields[1]) not in pairs:
            problems.append(f'{error_id}: "{fields[1]}" is not an error type under {fields[0]}')
        if fields[2] != "start" and fields[2] not in run_input["action_ids"]:
            problems.append(f"{error_id}: step {fields[2]} is not an action in this run")
    return problems

def call_llm(model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": MAX_TOKENS,
    }).encode()
    headers = {"Authorization": f"Bearer {KEY_FILE.read_text().strip()}", "Content-Type": "application/json"}
    for wait in (10, 30, 60, 120, 300):
        try:
            with urllib.request.urlopen(urllib.request.Request(API_URL, data=body, headers=headers), timeout=900) as response:
                reply = json.loads(response.read())
            return reply["choices"][0]["message"]["content"], reply["usage"]
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="ignore")
            if "insufficient_quota" in message or (error.code != 429 and error.code < 500):
                raise RuntimeError(f"API error {error.code}: {message}") from error
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(wait)
    raise RuntimeError("the API kept failing after 5 attempts")

def parse_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

def judge(run_input):
    problems = None
    usage = []
    for attempt in range(MAX_RETRIES + 1):
        text, used = call_llm(MODEL, build_prompt(run_input, problems))
        usage.append(used)
        answer = parse_json(text)
        problems = check(answer, run_input)
        if not problems:
            return answer, usage
    return {"failed": problems, "last_answer": answer}, usage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("findings_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", help="only judge these run ids")
    parser.add_argument("--limit", type=int, help="judge at most this many runs")
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt and its size, call nothing")
    args = parser.parse_args()
    rows = json.loads((args.results_dir / "baseline_results.json").read_text())
    findings = {}
    for finding in csv.DictReader(args.findings_csv.open()):
        findings.setdefault(finding["run_id"], []).append(finding)
    done = {json.loads(line)["run_id"] for line in args.out.open()} if args.out.exists() else set()
    todo = [row for row in rows if row["id"] in findings and row["id"] not in done and (not args.runs or row["id"] in args.runs)]
    for row in todo[: args.limit]:
        run_input = load_run(row, findings[row["id"]])
        if args.dry_run:
            prompt = build_prompt(run_input)
            print(prompt)
            print(f"\n[{row['id']}: {len(prompt):,} characters, about {len(prompt) // 4:,} tokens]")
            return
        result, usage = judge(run_input)
        with args.out.open("a") as out:
            out.write(json.dumps({"run_id": row["id"], "model": MODEL, "result": result, "usage": usage}) + "\n")
        print(f"{row['id']}: {'failed' if 'failed' in result else 'ok'}", flush=True)

if __name__ == "__main__":
    main()
