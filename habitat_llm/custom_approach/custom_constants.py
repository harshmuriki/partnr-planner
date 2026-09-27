from typing import Dict, List

CUSTOM_ACTIONS = "\n".join(
    (
        "Navigate: Navigate <agent_index> <entity_name>",
        "Explore: Explore <agent_index> <room_name>",
        "Open: Open <agent_index> <entity_name>",
        "Close: Close <agent_index> <entity_name>",
        "Pick: Pick <agent_index> <entity_name>",
        "Place: Place <agent_index> <entity_name_0,relation_0,entity_name_1,relation_1,entity_name_2>. Eg (Place 0 cup_0,on,counter_22,None,None)",
        "Fill: Fill <agent_index> <entity_name>",
        "Pour: Pour <agent_index> <entity_name>. Eg (Pour 0 cup_1) Should pour from the container already held by the agent into cup_1.",
        "Clean: Clean <agent_index> <entity_name>. Some objects require being near a faucet to clean.",
        "PowerOn: PowerOn <agent_index> <entity_name>",
        "PowerOff: PowerOff <agent_index> <entity_name>",
    )
)
CUSTOM_EXPLORATION_ACTIONS = "\n".join(
    (
        "Navigate: Navigate <agent_index> <entity_name>",
        "Explore: Explore <agent_index> <room_name>",
        "Open: Open <agent_index> <entity_name>",
    )
)
CUSTOM_ACTION_NAMES = (
    "Navigate",
    "Explore",
    "Open",
    "Close",
    "Pick",
    "Place",
    "Fill",
    "Pour",
    "Clean",
    "PowerOn",
    "PowerOff",
)
OBJECT_ALIAS_HINTS: Dict[str, List[str]] = {
    "towel": ["hand_towel_5"],
}

