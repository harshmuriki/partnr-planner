#!/usr/bin/env python3
"""Run custom_approach prompts through the OpenAI API.

Run from the repo root:
    PYTHONPATH=. python scripts/test_custom_approach_prompts.py

The script loads OPENAI_API_KEY from the repo-root .env file if it is not
already set in the shell environment.
"""

import argparse
import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from openai import OpenAI

from habitat_llm.custom_approach import prompt_builders
from habitat_llm.custom_approach import prompt_templates


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromptCase:
    name: str
    build: Callable[[], str]


SAMPLE_TASK = (
    "Heat some bread and place it in the bedroom you started at. Wash the bottles and place a towel next to them. "
    "And turn off the lights in the room you took the bread to and clean the towel."
)
SAMPLE_SUBGOAL = "Heat bread and place it in the bedroom you started at."
ACTIONS = (
    "Navigate: Navigate <agent_index> <entity_name>"
    "Explore: Explore <agent_index> <room_name>"
    "Open: Open <agent_index> <entity_name>"
    "Close: Close <agent_index> <entity_name>"
    "Pick: Pick <agent_index> <entity_name>"
    "Place: Place <agent_index> <entity_name_0,relation_0,entity_name_1,relation_1,entity_name_2>. Eg (Place 0 cup_0,on,counter_22,None,None)"
    "Fill: Fill <agent_index> <entity_name>"
    "Pour: Pour <agent_index> <entity_name>. Eg (Pour 0 cup_1) Should pour from the container already held by the agent into cup_1."
    "Clean: Clean <agent_index> <entity_name>. Some objects require being near a faucet to clean."
    "PowerOn: PowerOn <agent_index> <entity_name>"
    "PowerOff: PowerOff <agent_index> <entity_name>"
)
ACTION_NAMES = (
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

SAMPLE_ACTION_SEQUENCE = (
    "Navigate[0,bedroom_1]",
    "Pick[0,bread_1]",
    "Navigate[0,microwave_1]",
    "Place[0,bread_1,on,microwave_1,None,None]",
    "Navigate[0,bedroom_1]",
    "Pick[0,bottle_1]",
    "Navigate[0,bottle_2]",
)

SAMPLE_SCENE_GRAPH = """Hierarchical Scene Graph:
    Room: bathroom_1
            Furniture: floor_bathroom_1
            Furniture: toilet_52
                    Receptacle: rec_toilet_52_0
    Room: bathroom_2
            Furniture: floor_bathroom_2
            Furniture: toilet_54
                    Receptacle: rec_toilet_54_0
    Room: bathroom_3
            Furniture: floor_bathroom_3
            Furniture: shelves_43
                    Receptacle: rec_shelves_43_0
                    Receptacle: rec_shelves_43_1
                    Receptacle: rec_shelves_43_2
            Furniture: shelves_44
                    Receptacle: rec_shelves_44_0
                    Receptacle: rec_shelves_44_1
                    Receptacle: rec_shelves_44_2
            Furniture: toilet_53
                    Receptacle: rec_toilet_53_0
    Room: bedroom_1
            SpotRobot: agent_0
            Furniture: bed_57
                    Receptacle: rec_bed_57_0
            Furniture: chair_67
                    Receptacle: rec_chair_67_0
            Furniture: chest_of_drawers_79
                    Receptacle: rec_chest_of_drawers_79_0
                    Receptacle: rec_chest_of_drawers_79_1
            Furniture: chest_of_drawers_80
                    Receptacle: rec_chest_of_drawers_80_0
                    Receptacle: rec_chest_of_drawers_80_1
            Furniture: floor_bedroom_1
            Furniture: table_42
                    Receptacle: rec_table_42_0
            Furniture: table_96
                    Receptacle: rec_table_96_0
                    Receptacle: rec_table_96_1
                    Receptacle: rec_table_96_2
                    Receptacle: rec_table_96_3
            Furniture: unknown_87
                    Receptacle: rec_unknown_87_0
                    Receptacle: rec_unknown_87_1
                    Receptacle: rec_unknown_87_2
                    Receptacle: rec_unknown_87_3
                    Receptacle: rec_unknown_87_4
                    Receptacle: rec_unknown_87_5
            Furniture: wardrobe_112
                    Receptacle: rec_wardrobe_112_0
                    Receptacle: rec_wardrobe_112_1
            Furniture: wardrobe_113
                    Receptacle: rec_wardrobe_113_0
            Furniture: wardrobe_114
                    Receptacle: rec_wardrobe_114_0
    Room: bedroom_2
            Furniture: chair_51
                    Receptacle: rec_chair_51_0
                    Receptacle: rec_chair_51_1
            Furniture: chair_58
                    Receptacle: rec_chair_58_0
            Furniture: chair_61
                    Receptacle: rec_chair_61_0
            Furniture: chair_62
                    Receptacle: rec_chair_62_0
            Furniture: counter_90
                    Receptacle: rec_counter_90_0
            Furniture: floor_bedroom_2
            Furniture: shelves_23
                    Receptacle: rec_shelves_23_0
                    Receptacle: rec_shelves_23_1
                    Receptacle: rec_shelves_23_2
            Furniture: table_50
                    Receptacle: rec_table_50_0
            Furniture: table_60
                    Receptacle: rec_table_60_0
                    Receptacle: rec_table_60_1
    Room: bedroom_3
            Furniture: bed_21
                    Receptacle: rec_bed_21_0
                    Receptacle: rec_bed_21_1
                    Receptacle: rec_bed_21_2
                    Receptacle: rec_bed_21_3
            Furniture: bench_55
                    Receptacle: rec_bench_55_0
            Furniture: cabinet_92
                    Receptacle: rec_cabinet_92_0
            Furniture: chair_40
                    Receptacle: rec_chair_40_0
                    Receptacle: rec_chair_40_1
            Furniture: chair_70
                    Receptacle: rec_chair_70_0
            Furniture: chest_of_drawers_105
                    Receptacle: rec_chest_of_drawers_105_0
                    Receptacle: rec_chest_of_drawers_105_1
            Furniture: chest_of_drawers_106
                    Receptacle: rec_chest_of_drawers_106_0
                    Receptacle: rec_chest_of_drawers_106_1
            Furniture: floor_bedroom_3
            Furniture: table_115
                    Receptacle: rec_table_115_0
                    Receptacle: rec_table_115_1
            Furniture: table_56
                    Receptacle: rec_table_56_0
    Room: closet_1
    Room: closet_2
            Furniture: unknown_20
                    Receptacle: rec_unknown_20_0
                    Receptacle: rec_unknown_20_1
    Room: dining_room_1
            Furniture: cabinet_84
                    Receptacle: rec_cabinet_84_0
                    Receptacle: rec_cabinet_84_1
            Furniture: cabinet_85
                    Receptacle: rec_cabinet_85_0
                    Receptacle: rec_cabinet_85_1
            Furniture: floor_dining_room_1
            Furniture: table_66
                    Receptacle: rec_table_66_0
                    Receptacle: rec_table_66_1
                    Receptacle: rec_table_66_10
                    Receptacle: rec_table_66_11
                    Receptacle: rec_table_66_12
                    Receptacle: rec_table_66_13
                    Receptacle: rec_table_66_14
                    Receptacle: rec_table_66_15
                    Receptacle: rec_table_66_16
                    Receptacle: rec_table_66_17
                    Receptacle: rec_table_66_18
                    Receptacle: rec_table_66_19
                    Receptacle: rec_table_66_2
                    Receptacle: rec_table_66_20
                    Receptacle: rec_table_66_21
                    Receptacle: rec_table_66_22
                    Receptacle: rec_table_66_23
                    Receptacle: rec_table_66_24
                    Receptacle: rec_table_66_3
                    Receptacle: rec_table_66_4
                    Receptacle: rec_table_66_5
                    Receptacle: rec_table_66_6
                    Receptacle: rec_table_66_7
                    Receptacle: rec_table_66_8
                    Receptacle: rec_table_66_9
    Room: entryway_1
            Human: agent_1
            Furniture: chest_of_drawers_103
                    Receptacle: rec_chest_of_drawers_103_0
                    Receptacle: rec_chest_of_drawers_103_1
                    Receptacle: rec_chest_of_drawers_103_2
            Furniture: floor_entryway_1
            Furniture: table_22
                    Receptacle: rec_table_22_0
    Room: garage_1
            Furniture: cabinet_109
                    Receptacle: rec_cabinet_109_0
                    Receptacle: rec_cabinet_109_1
            Furniture: cabinet_110
                    Receptacle: rec_cabinet_110_0
                    Receptacle: rec_cabinet_110_1
            Furniture: cabinet_93
                    Receptacle: rec_cabinet_93_0
            Furniture: cabinet_94
                    Receptacle: rec_cabinet_94_0
            Furniture: cabinet_95
                    Receptacle: rec_cabinet_95_0
            Furniture: floor_garage_1
            Furniture: fridge_82
                    Receptacle: rec_fridge_82_0
                    Receptacle: rec_fridge_82_1
                    Receptacle: rec_fridge_82_2
                    Receptacle: rec_fridge_82_3
                    Receptacle: rec_fridge_82_4
            Furniture: shelves_28
                    Receptacle: rec_shelves_28_0
                    Receptacle: rec_shelves_28_1
            Furniture: shelves_29
                    Receptacle: rec_shelves_29_0
                    Receptacle: rec_shelves_29_1
    Room: hallway_1
            Furniture: floor_hallway_1
    Room: hallway_2
            Furniture: floor_hallway_2
            Furniture: shelves_32
                    Receptacle: rec_shelves_32_0
                    Receptacle: rec_shelves_32_1
    Room: kitchen_1
            Furniture: cabinet_100
                    Receptacle: rec_cabinet_100_0
                    Receptacle: rec_cabinet_100_1
            Furniture: cabinet_102
                    Receptacle: rec_cabinet_102_0
                    Receptacle: rec_cabinet_102_1
            Furniture: cabinet_71
                    Receptacle: rec_cabinet_71_0
            Furniture: cabinet_72
                    Receptacle: rec_cabinet_72_0
                    Receptacle: rec_cabinet_72_1
            Furniture: cabinet_73
                    Receptacle: rec_cabinet_73_0
                    Receptacle: rec_cabinet_73_1
            Furniture: cabinet_74
                    Receptacle: rec_cabinet_74_0
                    Receptacle: rec_cabinet_74_1
            Furniture: cabinet_75
                    Receptacle: rec_cabinet_75_0
                    Receptacle: rec_cabinet_75_1
            Furniture: cabinet_76
                    Receptacle: rec_cabinet_76_0
                    Receptacle: rec_cabinet_76_1
            Furniture: cabinet_77
                    Receptacle: rec_cabinet_77_0
                    Receptacle: rec_cabinet_77_1
            Furniture: cabinet_78
                    Receptacle: rec_cabinet_78_0
            Furniture: cabinet_91
                    Receptacle: rec_cabinet_91_0
                    Receptacle: rec_cabinet_91_1
            Furniture: cabinet_97
                    Receptacle: rec_cabinet_97_0
                    Receptacle: rec_cabinet_97_1
            Furniture: cabinet_98
                    Receptacle: rec_cabinet_98_0
                    Receptacle: rec_cabinet_98_1
            Furniture: cabinet_99
                    Receptacle: rec_cabinet_99_0
                    Receptacle: rec_cabinet_99_1
            Furniture: counter_24
                    Receptacle: rec_counter_24_0
            Furniture: counter_25
                    Receptacle: rec_counter_25_0
            Furniture: counter_26
                    Receptacle: rec_counter_26_0
            Furniture: counter_27
                    Receptacle: rec_counter_27_0
            Furniture: counter_88
                    Receptacle: rec_counter_88_0
                    Receptacle: rec_counter_88_1
            Furniture: counter_89
                    Receptacle: rec_counter_89_0
                    Receptacle: rec_counter_89_1
            Furniture: floor_kitchen_1
            Furniture: microwave_108
                    Receptacle: rec_microwave_108_0
                    Receptacle: rec_microwave_108_1
            Furniture: unknown_107
                    Receptacle: rec_unknown_107_0
                    Receptacle: rec_unknown_107_1
                    Receptacle: rec_unknown_107_2
                    Receptacle: rec_unknown_107_3
                    Receptacle: rec_unknown_107_4
                    Receptacle: rec_unknown_107_5
    Room: laundryroom_1
            Furniture: cabinet_101
                    Receptacle: rec_cabinet_101_0
                    Receptacle: rec_cabinet_101_1
            Furniture: cabinet_116
                    Receptacle: rec_cabinet_116_0
                    Receptacle: rec_cabinet_116_1
            Furniture: cabinet_117
                    Receptacle: rec_cabinet_117_0
                    Receptacle: rec_cabinet_117_1
            Furniture: floor_laundryroom_1
    Room: living_room_1
            Furniture: chair_30
                    Receptacle: rec_chair_30_0
                    Receptacle: rec_chair_30_1
                    Receptacle: rec_chair_30_2
            Furniture: chair_31
                    Receptacle: rec_chair_31_0
                    Receptacle: rec_chair_31_1
                    Receptacle: rec_chair_31_2
            Furniture: chair_65
                    Receptacle: rec_chair_65_0
            Furniture: chest_of_drawers_119
                    Receptacle: rec_chest_of_drawers_119_0
                    Receptacle: rec_chest_of_drawers_119_1
            Furniture: couch_33
                    Receptacle: rec_couch_33_0
                    Receptacle: rec_couch_33_1
                    Receptacle: rec_couch_33_2
            Furniture: couch_59
                    Receptacle: rec_couch_59_0
                    Receptacle: rec_couch_59_1
            Furniture: counter_83
                    Receptacle: rec_counter_83_0
                    Receptacle: rec_counter_83_1
                    Receptacle: rec_counter_83_2
            Furniture: floor_living_room_1
            Furniture: shelves_41
                    Receptacle: rec_shelves_41_0
                    Receptacle: rec_shelves_41_1
            Furniture: shelves_45
                    Receptacle: rec_shelves_45_0
                    Receptacle: rec_shelves_45_1
            Furniture: shelves_46
                    Receptacle: rec_shelves_46_0
                    Receptacle: rec_shelves_46_1
            Furniture: table_104
                    Receptacle: rec_table_104_0
                    Receptacle: rec_table_104_1
            Furniture: table_48
                    Receptacle: rec_table_48_0
            Furniture: table_49
                    Receptacle: rec_table_49_0
    Room: office_1
            Furniture: couch_63
                    Receptacle: rec_couch_63_0
                    Receptacle: rec_couch_63_1
                    Receptacle: rec_couch_63_2
                    Receptacle: rec_couch_63_3
                    Receptacle: rec_couch_63_4
                    Receptacle: rec_couch_63_5
            Furniture: counter_86
                    Receptacle: rec_counter_86_0
            Furniture: floor_office_1
            Furniture: stand_118
                    Receptacle: rec_stand_118_0
            Furniture: table_81
                    Receptacle: rec_table_81_0
                    Receptacle: rec_table_81_1
            Furniture: unknown_111
                    Receptacle: rec_unknown_111_0
                    Receptacle: rec_unknown_111_1
    Room: other_room_1
            Furniture: floor_other_room_1
            Furniture: shelves_34
                    Receptacle: rec_shelves_34_0
            Furniture: shelves_35
                    Receptacle: rec_shelves_35_0
            Furniture: shelves_36
                    Receptacle: rec_shelves_36_0
            Furniture: shelves_68
                    Receptacle: rec_shelves_68_0
            Furniture: unknown_47
                    Receptacle: rec_unknown_47_0
                    Receptacle: rec_unknown_47_1
    Room: outdoor_1
            Furniture: floor_outdoor_1
    Room: outdoor_2
            Furniture: floor_outdoor_2
    Room: porch_1
            Furniture: floor_porch_1
    Room: unknown_room
            Furniture: shelves_37
                    Receptacle: rec_shelves_37_0
            Furniture: shelves_38
                    Receptacle: rec_shelves_38_0
            Furniture: shelves_39
                    Receptacle: rec_shelves_39_0
            Furniture: shelves_69
                    Receptacle: rec_shelves_69_0
    Room: utilityroom_1
            Furniture: floor_utilityroom_1
            Furniture: shelves_64
                    Receptacle: rec_shelves_64_0

"""

SAMPLE_SCENE_GRAPH_AFTER_EXPLORATION = """Hierarchical Scene Graph:
    Room: bathroom_1
            Furniture: floor_bathroom_1
            Furniture: toilet_52
                    Receptacle: rec_toilet_52_0
    Room: bathroom_2
            Furniture: floor_bathroom_2
            Furniture: toilet_54
                    Receptacle: rec_toilet_54_0
    Room: bathroom_3
            Furniture: floor_bathroom_3
            Furniture: shelves_43
                    Receptacle: rec_shelves_43_0
                    Receptacle: rec_shelves_43_1
                    Receptacle: rec_shelves_43_2
                            Object: bottle_0
            Furniture: shelves_44
                    Receptacle: rec_shelves_44_0
                    Receptacle: rec_shelves_44_1
                    Receptacle: rec_shelves_44_2
            Furniture: toilet_53
                    Receptacle: rec_toilet_53_0
    Room: bedroom_1
            Furniture: bed_57
                    Receptacle: rec_bed_57_0
            Furniture: chair_67
                    Receptacle: rec_chair_67_0
            Furniture: chest_of_drawers_79
                    Receptacle: rec_chest_of_drawers_79_0
                            Object: lamp_2
                    Receptacle: rec_chest_of_drawers_79_1
            Furniture: chest_of_drawers_80
                    Receptacle: rec_chest_of_drawers_80_0
                            Object: lamp_3
                    Receptacle: rec_chest_of_drawers_80_1
            Furniture: floor_bedroom_1
            Furniture: table_42
                    Receptacle: rec_table_42_0
                            Object: bottle_10
            Furniture: table_96
                    Receptacle: rec_table_96_0
                    Receptacle: rec_table_96_1
                            Object: bottle_11
                    Receptacle: rec_table_96_2
                    Receptacle: rec_table_96_3
            Furniture: unknown_87
                    Receptacle: rec_unknown_87_0
                    Receptacle: rec_unknown_87_1
                    Receptacle: rec_unknown_87_2
                    Receptacle: rec_unknown_87_3
                    Receptacle: rec_unknown_87_4
                    Receptacle: rec_unknown_87_5
            Furniture: wardrobe_112
                    Receptacle: rec_wardrobe_112_0
                    Receptacle: rec_wardrobe_112_1
            Furniture: wardrobe_113
                    Receptacle: rec_wardrobe_113_0
            Furniture: wardrobe_114
                    Receptacle: rec_wardrobe_114_0
    Room: bedroom_2
            Furniture: chair_51
                    Receptacle: rec_chair_51_0
                    Receptacle: rec_chair_51_1
            Furniture: chair_58
                    Receptacle: rec_chair_58_0
            Furniture: chair_61
                    Receptacle: rec_chair_61_0
            Furniture: chair_62
                    Receptacle: rec_chair_62_0
            Furniture: counter_90
                    Receptacle: rec_counter_90_0
            Furniture: floor_bedroom_2
            Furniture: shelves_23
                    Receptacle: rec_shelves_23_0
                    Receptacle: rec_shelves_23_1
                    Receptacle: rec_shelves_23_2
            Furniture: table_50
                    Receptacle: rec_table_50_0
            Furniture: table_60
                    Receptacle: rec_table_60_0
                    Receptacle: rec_table_60_1
    Room: bedroom_3
            SpotRobot: agent_0
            Furniture: bed_21
                    Receptacle: rec_bed_21_0
                    Receptacle: rec_bed_21_1
                    Receptacle: rec_bed_21_2
                    Receptacle: rec_bed_21_3
            Furniture: bench_55
                    Receptacle: rec_bench_55_0
            Furniture: cabinet_92
                    Receptacle: rec_cabinet_92_0
            Furniture: chair_40
                    Receptacle: rec_chair_40_0
                    Receptacle: rec_chair_40_1
            Furniture: chair_70
                    Receptacle: rec_chair_70_0
            Furniture: chest_of_drawers_105
                    Receptacle: rec_chest_of_drawers_105_0
                    Receptacle: rec_chest_of_drawers_105_1
            Furniture: chest_of_drawers_106
                    Receptacle: rec_chest_of_drawers_106_0
                    Receptacle: rec_chest_of_drawers_106_1
            Furniture: floor_bedroom_3
            Furniture: table_115
                    Receptacle: rec_table_115_0
                    Receptacle: rec_table_115_1
            Furniture: table_56
                    Receptacle: rec_table_56_0
    Room: closet_1
    Room: closet_2
            Furniture: unknown_20
                    Receptacle: rec_unknown_20_0
                    Receptacle: rec_unknown_20_1
    Room: dining_room_1
            Furniture: cabinet_84
                    Receptacle: rec_cabinet_84_0
                    Receptacle: rec_cabinet_84_1
            Furniture: cabinet_85
                    Receptacle: rec_cabinet_85_0
                    Receptacle: rec_cabinet_85_1
            Furniture: floor_dining_room_1
            Furniture: table_66
                    Receptacle: rec_table_66_0
                    Receptacle: rec_table_66_1
                    Receptacle: rec_table_66_10
                    Receptacle: rec_table_66_11
                    Receptacle: rec_table_66_12
                    Receptacle: rec_table_66_13
                    Receptacle: rec_table_66_14
                    Receptacle: rec_table_66_15
                    Receptacle: rec_table_66_16
                    Receptacle: rec_table_66_17
                    Receptacle: rec_table_66_18
                    Receptacle: rec_table_66_19
                            Object: cup_6
                    Receptacle: rec_table_66_2
                    Receptacle: rec_table_66_20
                    Receptacle: rec_table_66_21
                    Receptacle: rec_table_66_22
                    Receptacle: rec_table_66_23
                    Receptacle: rec_table_66_24
                            Object: bread_4
                            Object: plate_8
                    Receptacle: rec_table_66_3
                    Receptacle: rec_table_66_4
                    Receptacle: rec_table_66_5
                    Receptacle: rec_table_66_6
                    Receptacle: rec_table_66_7
                    Receptacle: rec_table_66_8
                    Receptacle: rec_table_66_9
                            Object: bottle_9
    Room: entryway_1
            Human: agent_1
            Furniture: chest_of_drawers_103
                    Receptacle: rec_chest_of_drawers_103_0
                    Receptacle: rec_chest_of_drawers_103_1
                    Receptacle: rec_chest_of_drawers_103_2
            Furniture: floor_entryway_1
            Furniture: table_22
                    Receptacle: rec_table_22_0
    Room: garage_1
            Furniture: cabinet_109
                    Receptacle: rec_cabinet_109_0
                    Receptacle: rec_cabinet_109_1
            Furniture: cabinet_110
                    Receptacle: rec_cabinet_110_0
                    Receptacle: rec_cabinet_110_1
            Furniture: cabinet_93
                    Receptacle: rec_cabinet_93_0
            Furniture: cabinet_94
                    Receptacle: rec_cabinet_94_0
            Furniture: cabinet_95
                    Receptacle: rec_cabinet_95_0
            Furniture: floor_garage_1
            Furniture: fridge_82
                    Receptacle: rec_fridge_82_0
                    Receptacle: rec_fridge_82_1
                    Receptacle: rec_fridge_82_2
                    Receptacle: rec_fridge_82_3
                    Receptacle: rec_fridge_82_4
            Furniture: shelves_28
                    Receptacle: rec_shelves_28_0
                            Object: can_7
                            Object: lamp_1
                    Receptacle: rec_shelves_28_1
            Furniture: shelves_29
                    Receptacle: rec_shelves_29_0
                    Receptacle: rec_shelves_29_1
    Room: hallway_1
            Furniture: floor_hallway_1
    Room: hallway_2
            Furniture: floor_hallway_2
            Furniture: shelves_32
                    Receptacle: rec_shelves_32_0
                    Receptacle: rec_shelves_32_1
    Room: kitchen_1
            Furniture: cabinet_100
                    Receptacle: rec_cabinet_100_0
                    Receptacle: rec_cabinet_100_1
            Furniture: cabinet_102
                    Receptacle: rec_cabinet_102_0
                    Receptacle: rec_cabinet_102_1
            Furniture: cabinet_71
                    Receptacle: rec_cabinet_71_0
            Furniture: cabinet_72
                    Receptacle: rec_cabinet_72_0
                    Receptacle: rec_cabinet_72_1
            Furniture: cabinet_73
                    Receptacle: rec_cabinet_73_0
                    Receptacle: rec_cabinet_73_1
            Furniture: cabinet_74
                    Receptacle: rec_cabinet_74_0
                    Receptacle: rec_cabinet_74_1
            Furniture: cabinet_75
                    Receptacle: rec_cabinet_75_0
                            Object: hand_towel_5
                    Receptacle: rec_cabinet_75_1
            Furniture: cabinet_76
                    Receptacle: rec_cabinet_76_0
                    Receptacle: rec_cabinet_76_1
            Furniture: cabinet_77
                    Receptacle: rec_cabinet_77_0
                    Receptacle: rec_cabinet_77_1
            Furniture: cabinet_78
                    Receptacle: rec_cabinet_78_0
            Furniture: cabinet_91
                    Receptacle: rec_cabinet_91_0
                    Receptacle: rec_cabinet_91_1
            Furniture: cabinet_97
                    Receptacle: rec_cabinet_97_0
                    Receptacle: rec_cabinet_97_1
            Furniture: cabinet_98
                    Receptacle: rec_cabinet_98_0
                    Receptacle: rec_cabinet_98_1
            Furniture: cabinet_99
                    Receptacle: rec_cabinet_99_0
                    Receptacle: rec_cabinet_99_1
            Furniture: counter_24
                    Receptacle: rec_counter_24_0
            Furniture: counter_25
                    Receptacle: rec_counter_25_0
            Furniture: counter_26
                    Receptacle: rec_counter_26_0
            Furniture: counter_27
                    Receptacle: rec_counter_27_0
            Furniture: counter_88
                    Receptacle: rec_counter_88_0
                    Receptacle: rec_counter_88_1
            Furniture: counter_89
                    Receptacle: rec_counter_89_0
                    Receptacle: rec_counter_89_1
            Furniture: floor_kitchen_1
            Furniture: microwave_108
                    Receptacle: rec_microwave_108_0
                    Receptacle: rec_microwave_108_1
            Furniture: unknown_107
                    Receptacle: rec_unknown_107_0
                    Receptacle: rec_unknown_107_1
                    Receptacle: rec_unknown_107_2
                    Receptacle: rec_unknown_107_3
                    Receptacle: rec_unknown_107_4
                    Receptacle: rec_unknown_107_5
    Room: laundryroom_1
            Furniture: cabinet_101
                    Receptacle: rec_cabinet_101_0
                    Receptacle: rec_cabinet_101_1
            Furniture: cabinet_116
                    Receptacle: rec_cabinet_116_0
                    Receptacle: rec_cabinet_116_1
            Furniture: cabinet_117
                    Receptacle: rec_cabinet_117_0
                    Receptacle: rec_cabinet_117_1
            Furniture: floor_laundryroom_1
    Room: living_room_1
            Furniture: chair_30
                    Receptacle: rec_chair_30_0
                    Receptacle: rec_chair_30_1
                    Receptacle: rec_chair_30_2
            Furniture: chair_31
                    Receptacle: rec_chair_31_0
                    Receptacle: rec_chair_31_1
                    Receptacle: rec_chair_31_2
            Furniture: chair_65
                    Receptacle: rec_chair_65_0
            Furniture: chest_of_drawers_119
                    Receptacle: rec_chest_of_drawers_119_0
                    Receptacle: rec_chest_of_drawers_119_1
            Furniture: couch_33
                    Receptacle: rec_couch_33_0
                    Receptacle: rec_couch_33_1
                    Receptacle: rec_couch_33_2
            Furniture: couch_59
                    Receptacle: rec_couch_59_0
                    Receptacle: rec_couch_59_1
            Furniture: counter_83
                    Receptacle: rec_counter_83_0
                    Receptacle: rec_counter_83_1
                    Receptacle: rec_counter_83_2
            Furniture: floor_living_room_1
            Furniture: shelves_41
                    Receptacle: rec_shelves_41_0
                    Receptacle: rec_shelves_41_1
            Furniture: shelves_45
                    Receptacle: rec_shelves_45_0
                    Receptacle: rec_shelves_45_1
            Furniture: shelves_46
                    Receptacle: rec_shelves_46_0
                    Receptacle: rec_shelves_46_1
            Furniture: table_104
                    Receptacle: rec_table_104_0
                    Receptacle: rec_table_104_1
            Furniture: table_48
                    Receptacle: rec_table_48_0
            Furniture: table_49
                    Receptacle: rec_table_49_0
    Room: office_1
            Furniture: couch_63
                    Receptacle: rec_couch_63_0
                    Receptacle: rec_couch_63_1
                    Receptacle: rec_couch_63_2
                    Receptacle: rec_couch_63_3
                    Receptacle: rec_couch_63_4
                    Receptacle: rec_couch_63_5
            Furniture: counter_86
                    Receptacle: rec_counter_86_0
            Furniture: floor_office_1
            Furniture: stand_118
                    Receptacle: rec_stand_118_0
            Furniture: table_81
                    Receptacle: rec_table_81_0
                    Receptacle: rec_table_81_1
            Furniture: unknown_111
                    Receptacle: rec_unknown_111_0
                    Receptacle: rec_unknown_111_1
    Room: other_room_1
            Furniture: floor_other_room_1
            Furniture: shelves_34
                    Receptacle: rec_shelves_34_0
            Furniture: shelves_35
                    Receptacle: rec_shelves_35_0
            Furniture: shelves_36
                    Receptacle: rec_shelves_36_0
            Furniture: shelves_68
                    Receptacle: rec_shelves_68_0
            Furniture: unknown_47
                    Receptacle: rec_unknown_47_0
                    Receptacle: rec_unknown_47_1
    Room: outdoor_1
            Furniture: floor_outdoor_1
    Room: outdoor_2
            Furniture: floor_outdoor_2
    Room: porch_1
            Furniture: floor_porch_1
    Room: unknown_room
            Furniture: shelves_37
                    Receptacle: rec_shelves_37_0
            Furniture: shelves_38
                    Receptacle: rec_shelves_38_0
            Furniture: shelves_39
                    Receptacle: rec_shelves_39_0
            Furniture: shelves_69
                    Receptacle: rec_shelves_69_0
    Room: utilityroom_1
            Furniture: floor_utilityroom_1
            Furniture: shelves_64
                    Receptacle: rec_shelves_64_0
"""


def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overwriting existing variables."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_cases() -> List[PromptCase]:
    return [
        PromptCase(
            name="task_to_object_subgoals_template",
            build=lambda: prompt_templates.TASK_TO_OBJECT_SUBGOALS_PROMPT.format(
                high_level_task=SAMPLE_TASK
            ),
        ),
        PromptCase(
            name="subgoal_exploration_template",
            build=lambda: prompt_templates.SUBGOAL_EXPLORATION_PROMPT.format(
                high_level_task=SAMPLE_TASK,
                sub_goal=SAMPLE_SUBGOAL,
                actions=ACTIONS,
                scene_graph=SAMPLE_SCENE_GRAPH,
                last_action_result="(none)",
                completed_objects="(none)",
                exploration_history="[]",
            ),
        ),
        PromptCase(
            name="subgoal_to_vlm_actions_template",
            build=lambda: prompt_templates.SUBGOAL_TO_VLM_ACTIONS_PROMPT.format(
                high_level_task=SAMPLE_TASK,
                sub_goal=SAMPLE_SUBGOAL,
                actions=ACTIONS,
                scene_graph=SAMPLE_SCENE_GRAPH_AFTER_EXPLORATION,
                action_history='["Explore[kitchen]"]',
            ),
        ),
    ]


def ask_openai(
    client: OpenAI,
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_completion_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a household task-planning assistant. Follow the "
                    "user prompt's output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    return response.choices[0].message.content or ""


def parse_python_list(raw_output: str) -> List[str]:
    try:
        parsed = ast.literal_eval(raw_output.strip())
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def is_action_like(item: str) -> bool:
    item = item.strip()
    return any(
        item == action_name
        or item.startswith(f"{action_name} ")
        or item.startswith(f"{action_name}[")
        for action_name in ACTION_NAMES
    )


def build_object_alias_hint(subgoal: str) -> str:
    subgoal_lower = subgoal.lower()
    matches = [
        f"- {term}: use {', '.join(aliases)} if present in the scene graph"
        for term, aliases in OBJECT_ALIAS_HINTS.items()
        if term in subgoal_lower
    ]
    if not matches:
        return ""
    return (
        "\n\n# Object Alias Grounding\n"
        "Use these scene-graph names when the subgoal uses a generic term:\n"
        + "\n".join(matches)
        + "\n"
    )


def print_section(title: str) -> None:
    print("=" * 88)
    print(title)
    print("=" * 88)


def maybe_print_prompt(prompt: str, print_prompts: bool) -> None:
    if print_prompts:
        # Keep prompts readable by hiding the large scene-graph body.
        prompt_for_display = re.sub(
            r"(Scene graph:\s*)([\s\S]*?)(\n(?:Explored rooms/furniture:|Action history:|# Instructions))",
            r"\1<omitted for readability>\3",
            prompt,
        )
        print("\nPROMPT:\n")
        print(prompt_for_display)
        print("\nOPENAI RESPONSE:\n")


def print_model_output(raw_output: str) -> None:
    """Pretty-print model output for readability."""
    parsed_list = parse_python_list(raw_output)
    if parsed_list:
        print("OPENAI RESPONSE (list):")
        for idx, item in enumerate(parsed_list, start=1):
            print(f"  {idx:02d}. {item}")
    else:
        print("OPENAI RESPONSE:")
        print(raw_output.strip())


def run_workflow(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    print_prompts: bool,
) -> None:
    """Run Task -> subgoals -> per-subgoal exploration + VLM actions."""
    subgoal_prompt = prompt_templates.TASK_TO_OBJECT_SUBGOALS_PROMPT.format(
        high_level_task=SAMPLE_TASK
    )
    print_section("STEP 1: Task -> Subgoals")
    maybe_print_prompt(subgoal_prompt, print_prompts)
    raw_subgoals = ask_openai(
        client,
        subgoal_prompt,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    print_model_output(raw_subgoals)
    print()

    subgoals = parse_python_list(raw_subgoals)
    if not subgoals:
        print("Could not parse subgoal output as a Python list. Stopping workflow.")
        return

    action_history: List[str] = []
    exploration_history: List[str] = []

    for idx, subgoal in enumerate(subgoals, start=1):
        print_section(f"SUBGOAL {idx}/{len(subgoals)}: {subgoal}")

        needed_objects: List[str] = []
        max_exploration_rounds = 3
        for explore_round in range(1, max_exploration_rounds + 1):
            current_scene_graph = (
                SAMPLE_SCENE_GRAPH_AFTER_EXPLORATION if explore_round > 2 else SAMPLE_SCENE_GRAPH # Adding uncertainty in the mix
            )
            exploration_prompt = (
                prompt_templates.SUBGOAL_EXPLORATION_PROMPT.format(
                    high_level_task=SAMPLE_TASK,
                    sub_goal=subgoal,
                    actions=ACTIONS,
                    scene_graph=current_scene_graph,
                    last_action_result=(
                        exploration_history[-1] if exploration_history else "(none)"
                    ),
                    completed_objects="(none)",
                    exploration_history=str(exploration_history),
                )
                + build_object_alias_hint(subgoal)
            )
            print_section(
                f"SUBGOAL {idx}: Exploration Round {explore_round}"
            )
            maybe_print_prompt(exploration_prompt, print_prompts)
            raw_exploration = ask_openai(
                client,
                exploration_prompt,
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            print_model_output(raw_exploration)
            print()

            exploration_output = parse_python_list(raw_exploration)
            if exploration_output and not any(
                is_action_like(item) for item in exploration_output
            ):
                needed_objects = exploration_output
                print("Exploration ended with needed objects:")
                for obj_idx, obj_name in enumerate(needed_objects, start=1):
                    print(f"  {obj_idx:02d}. {obj_name}")
                print()
                break

            if not exploration_output:
                print("Exploration did not return actions or objects; stopping this subgoal.")
                print()
                break

            exploration_history.extend(exploration_output)
            action_history.extend(exploration_output)

        if not needed_objects:
            print("No needed objects were found for this subgoal; skipping VLM actions.")
            print()
            continue
        
        ACTION_RETRIES = 3
        for action_retry in range(1, ACTION_RETRIES + 1):
            print_section(f"SUBGOAL {idx}: VLM Actions Round {action_retry}")
            vlm_actions_prompt = prompt_templates.SUBGOAL_TO_VLM_ACTIONS_PROMPT.format(
                high_level_task=SAMPLE_TASK,
                sub_goal=f"{subgoal}\nNeeded objects: {needed_objects}",
                actions=ACTIONS,
                scene_graph=SAMPLE_SCENE_GRAPH_AFTER_EXPLORATION,
                action_history=str(action_history),
                replan_needed="True" if action_retry <= ACTION_RETRIES else "False",
            ) + build_object_alias_hint(subgoal)
            maybe_print_prompt(vlm_actions_prompt, print_prompts)
            raw_vlm_actions = ask_openai(
                client,
                vlm_actions_prompt,
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            print_model_output(raw_vlm_actions)
            print()

            vlm_actions = parse_python_list(raw_vlm_actions)


            if action_retry <= 2:
                # simulate the error
                error = "The actions errored because the gripper couldn't pick/move the object"
                vlm_actions += [error]

            action_history.extend(vlm_actions)

def parse_args() -> argparse.Namespace:
    case_names = [case.name for case in build_cases()]
    parser = argparse.ArgumentParser(
        description="Send custom_approach prompts to OpenAI and print raw outputs."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-5.6-luna.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-completion-tokens", type=int, default=500)
    parser.add_argument(
        "--mode",
        choices=["workflow", "individual"],
        default="workflow",
        help=(
            "workflow runs Task -> subgoals -> per-subgoal exploration and VLM "
            "actions. individual runs the selected standalone prompt case."
        ),
    )
    parser.add_argument(
        "--case",
        choices=case_names + ["all"],
        default="all",
        help="Run one prompt case or all cases.",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print each formatted prompt before the OpenAI response.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required in the environment or .env.")

    client = OpenAI(api_key=api_key)

    if args.mode == "workflow":
        run_workflow(
            client,
            model=args.model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            print_prompts=args.print_prompts,
        )
        print("Done.")
        return

    cases = build_cases()
    if args.case != "all":
        cases = [case for case in cases if case.name == args.case]

    for idx, case in enumerate(cases, start=1):
        prompt = case.build()
        print_section(f"[{idx}/{len(cases)}] {case.name}")
        maybe_print_prompt(prompt, args.print_prompts)
        output = ask_openai(
            client,
            prompt,
            model=args.model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
        print_model_output(output)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
