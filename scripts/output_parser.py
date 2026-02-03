#!/usr/bin/env python3
"""
Parser for PARTNR planner trace files.
Parses trace files and displays them in a clear, readable format.
"""

import re
import sys
from typing import List, Dict, Optional
from pathlib import Path


class TraceParser:
    """Parser for trace files from PARTNR planner."""

    def __init__(self, trace_file: str):
        self.trace_file = Path(trace_file)
        self.task = ""
        self.steps: List[Dict] = []

    def parse(self):
        """Parse the trace file."""
        if not self.trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_file}")

        with open(self.trace_file, 'r') as f:
            content = f.read()

        # Extract task (first line)
        lines = content.split('\n')
        if lines[0].startswith('Task:'):
            self.task = lines[0].replace('Task:', '').strip()

        # Parse the rest of the content
        # The format is: Thought -> Action -> Result -> Objects
        # Sometimes they're on the same line, sometimes separate

        # Join all lines and split by patterns
        full_text = '\n'.join(lines[1:])  # Skip task line

        # Pattern to find steps: Thought: ... Action[...] Assigned!Result: ... Objects: ...
        # Split by "Thought:" markers
        thought_pattern = r'Thought:\s*(.*?)(?=Thought:|Action:|$|Assigned!)'
        action_pattern = r'([A-Za-z]+)\[([^\]]*)\]'
        result_pattern = r'Assigned!Result:\s*(.*?)(?=Objects:|Thought:|$)'
        objects_pattern = r'Objects:\s*(.*?)(?=Thought:|Action:|$)'

        # Find all thoughts
        thoughts = re.findall(thought_pattern, full_text, re.DOTALL)

        # Find all actions
        actions = re.findall(action_pattern, full_text)

        # Find all results
        results = re.findall(result_pattern, full_text, re.DOTALL)

        # Find all objects sections
        objects_sections = re.findall(objects_pattern, full_text, re.DOTALL)

        # Reconstruct steps by finding patterns in order
        # More robust: split by "Thought:" and parse each section
        sections = re.split(r'(?=Thought:)', full_text)

        step_num = 0
        for section in sections:
            if not section.strip():
                continue

            step = {}

            # Extract thought
            # Look for Thought: that might appear after objects (concatenated without newline)
            thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|Assigned!|Explore\[|Rearrange\[|Place\[|Navigate\[|Done\[|$)', section, re.DOTALL)
            if thought_match:
                thought_text = thought_match.group(1).strip()
                # Remove duplicate "Thought:" prefix if present
                if thought_text.startswith('Thought:'):
                    thought_text = thought_text.replace('Thought:', '', 1).strip()
                # Remove action that might be embedded in thought
                thought_text = re.sub(r'([A-Za-z]+)\[([^\]]*)\]', '', thought_text).strip()
                # Remove any object-like patterns that might have been captured
                thought_text = re.sub(r'^\d+_\w+:\s*.*?States:.*?', '', thought_text, flags=re.DOTALL).strip()
                if thought_text:
                    step['thought'] = thought_text

            # Extract action
            action_match = re.search(r'([A-Za-z]+)\[([^\]]*)\]', section)
            if action_match:
                step['action_name'] = action_match.group(1)
                step['action_args'] = action_match.group(2)

            # Extract result
            result_match = re.search(r'Assigned!Result:\s*(.*?)(?=Objects:|Thought:|$)', section, re.DOTALL)
            if result_match:
                result_text = result_match.group(1).strip()
                step['result'] = result_text
                step['success'] = 'Successful' in result_text or 'success' in result_text.lower()
                if 'Unexpected failure' in result_text or 'Failed' in result_text:
                    step['success'] = False
                    step['error'] = result_text.replace('Unexpected failure! -', '').strip()

            # Extract objects
            objects_match = re.search(r'Objects:\s*(.*?)(?=Thought:|Action:|$)', section, re.DOTALL)
            if objects_match:
                objects_text = objects_match.group(1).strip()
                # Remove "Thought:" if it appears at the end (concatenated without newline)
                objects_text = re.sub(r'Thought:\s*$', '', objects_text, flags=re.MULTILINE)
                objects_text = objects_text.strip()
                step['objects'] = self._parse_objects(objects_text)

            if step:
                step['step_num'] = step_num
                self.steps.append(step)
                step_num += 1

    def _parse_objects(self, objects_text: str) -> List[Dict]:
        """Parse objects text into structured format."""
        objects = []

        # Split by newlines and parse each object
        lines = objects_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove "Thought:" if it appears anywhere in the line (concatenated without newline)
            # This handles cases like: "object: location. States: ...Thought: ..."
            if 'Thought:' in line:
                # Split on "Thought:" and only keep the part before it
                line = line.split('Thought:')[0].strip()
            if not line:
                continue

            # Format: object_name: location. States: state1: value1, state2: value2
            # Or: object_name: held by the agent. States: ...
            # Or: object_name: held by the other agent. States: ...

            if ':' not in line:
                continue

            obj = {}

            # Extract object name and location
            parts = line.split(':')
            if len(parts) < 2:
                continue

            obj['name'] = parts[0].strip()

            # Location is everything after first colon until "States:"
            location_part = ':'.join(parts[1:])

            if 'States:' in location_part:
                location, states_part = location_part.split('States:', 1)
                obj['location'] = location.strip()

                # Parse states
                states = {}
                states_text = states_part.strip()
                if states_text:
                    # Format: state1: value1, state2: value2
                    # Handle "powered on" as a single state name (two words)
                    # Use regex to find all "key: value" pairs, handling multi-word keys
                    # Pattern: word(s) followed by colon and value
                    state_pattern = r'([a-z_]+(?:\s+[a-z_]+)?):\s*(\w+)'
                    state_matches = re.findall(state_pattern, states_text, re.IGNORECASE)
                    for key, value in state_matches:
                        key = key.strip()
                        value = value.strip()
                        # Convert string booleans
                        if value.lower() == 'true':
                            states[key] = True
                        elif value.lower() == 'false':
                            states[key] = False
                        else:
                            states[key] = value
                obj['states'] = states
            else:
                obj['location'] = location_part.strip()
                obj['states'] = {}

            objects.append(obj)

        return objects

    def display(self, output_file: Optional[str] = None):
        """Display the parsed trace in a readable format."""
        output_lines = []

        # Header
        output_lines.append("=" * 80)
        output_lines.append("TASK")
        output_lines.append("=" * 80)
        output_lines.append(self.task)
        output_lines.append("")

        # Steps
        for step in self.steps:
            output_lines.append("=" * 80)
            output_lines.append(f"STEP {step['step_num'] + 1}")
            output_lines.append("=" * 80)

            # Thought
            if 'thought' in step:
                output_lines.append("💭 THOUGHT:")
                thought_lines = step['thought'].split('\n')
                for line in thought_lines:
                    output_lines.append(f"   {line.strip()}")
                output_lines.append("")

            # Action
            if 'action_name' in step:
                action_str = f"{step['action_name']}[{step['action_args']}]"
                output_lines.append(f"⚡ ACTION: {action_str}")
                output_lines.append("")

            # Result
            if 'result' in step:
                status = "✅ SUCCESS" if step.get('success', False) else "❌ FAILURE"
                output_lines.append(f"{status}:")
                if step.get('error'):
                    output_lines.append(f"   Error: {step['error']}")
                else:
                    output_lines.append(f"   {step['result']}")
                output_lines.append("")

            # Objects
            if 'objects' in step and step['objects']:
                output_lines.append("📦 OBJECTS:")
                for obj in step['objects']:
                    location = obj.get('location', 'unknown')
                    states_str = ""
                    if obj.get('states'):
                        states_list = [f"{k}={v}" for k, v in obj['states'].items()]
                        states_str = f" | States: {', '.join(states_list)}"
                    output_lines.append(f"   • {obj['name']}: {location}{states_str}")
                output_lines.append("")

        # Footer
        output_lines.append("=" * 80)
        output_lines.append(f"Total Steps: {len(self.steps)}")
        output_lines.append("=" * 80)

        output_text = '\n'.join(output_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_text)
            print(f"Output written to: {output_file}")
        else:
            print(output_text)

        return output_text


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python output_parser.py <trace_file> [output_file]")
        print("Example: python output_parser.py trace-episode_3_0-1.txt parsed_output.txt")
        sys.exit(1)

    trace_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    parser = TraceParser(trace_file)
    parser.parse()
    parser.display(output_file)


if __name__ == '__main__':
    main()
