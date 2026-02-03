#!/usr/bin/env python3
"""
HTML viewer for PARTNR trace logs.
Converts trace text files to an interactive HTML page.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Any


def parse_trace_file(trace_file: str) -> Dict[str, Any]:
    """Parse a trace text file into structured data."""
    with open(trace_file, 'r') as f:
        content = f.read()

    # Extract task (first line)
    task = ""
    lines = content.split('\n')
    if lines and lines[0].startswith('Task:'):
        task = lines[0].replace('Task:', '').strip()

    # Parse steps by splitting on "Thought:" markers
    steps = []

    # Split content by "Thought:" to find each step
    sections = re.split(r'(?=Thought:)', content)

    for section in sections:
        if not section.strip() or section.strip().startswith('Task:'):
            continue

        step = {}

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n(?:[A-Z][a-z]+\[|Assigned!|$))', section, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            # Remove nested "Thought:" if present
            thought = re.sub(r'^Thought:\s*', '', thought).strip()
            step['thought'] = thought

        # Extract action (Action[args] format)
        action_match = re.search(r'([A-Z][a-z]+)\[([^\]]*)\]', section)
        if action_match:
            step['action'] = action_match.group(1)
            step['args'] = action_match.group(2)

        # Extract result
        result_match = re.search(r'Assigned!Result:\s*(.*?)(?=\nObjects:|Thought:|$)', section, re.DOTALL)
        if result_match:
            result = result_match.group(1).strip()
            step['result'] = result
            step['success'] = 'Successful' in result or 'success' in result.lower()

        # Extract objects (all lines between "Objects:" and next "Thought:")
        objects_match = re.search(r'Objects:\s*(.*?)(?=Thought:|$)', section, re.DOTALL)
        if objects_match:
            objects_text = objects_match.group(1).strip()
            if objects_text and objects_text != 'No objects found yet':
                step['objects'] = objects_text

        if step:
            steps.append(step)

    return {
        'task': task,
        'steps': steps,
        'total_steps': len(steps)
    }


def generate_html(trace_data: Dict[str, Any], output_file: str) -> None:
    """Generate an HTML file from parsed trace data."""

    task = trace_data.get('task', 'No task')
    steps = trace_data.get('steps', [])
    total_steps = trace_data.get('total_steps', 0)

    # Count successes
    successes = sum(1 for s in steps if s.get('success', False))
    success_rate = (successes / total_steps * 100) if total_steps > 0 else 0

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trace Log Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 8px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-label {{
            font-size: 12px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .task-section {{
            background: #f3f4f6;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 30px;
        }}
        
        .task-section h2 {{
            font-size: 16px;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .task-section p {{
            font-size: 18px;
            color: #1f2937;
            line-height: 1.6;
        }}
        
        .controls {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.3s ease;
        }}
        
        .btn:hover {{
            background: #5568d3;
        }}
        
        .filter-btn {{
            background: #e5e7eb;
            color: #374151;
        }}
        
        .filter-btn:hover {{
            background: #d1d5db;
        }}
        
        .filter-btn.active {{
            background: #10b981;
            color: white;
        }}
        
        .steps {{
            margin-top: 20px;
        }}
        
        .step {{
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .step.success {{
            border-left: 4px solid #10b981;
        }}
        
        .step.failure {{
            border-left: 4px solid #ef4444;
        }}
        
        .step:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .step-header {{
            background: linear-gradient(to right, #f3f4f6, #e5e7eb);
            padding: 15px 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            cursor: pointer;
            user-select: none;
        }}
        
        .step-number {{
            background: #667eea;
            color: white;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            flex-shrink: 0;
        }}
        
        .step-action {{
            flex: 1;
            font-family: 'Monaco', 'Courier New', monospace;
            font-weight: bold;
            color: #1f2937;
            font-size: 15px;
        }}
        
        .step-status {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            flex-shrink: 0;
        }}
        
        .status-success {{
            background: #d1fae5;
            color: #065f46;
        }}
        
        .status-failure {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .step-content {{
            padding: 20px;
            display: none;
            border-top: 1px solid #e5e7eb;
        }}
        
        .step.expanded .step-content {{
            display: block;
        }}
        
        .section {{
            margin-bottom: 20px;
        }}
        
        .section-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #667eea;
            margin-bottom: 8px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .section-content {{
            padding: 15px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        }}
        
        .thought {{
            font-style: italic;
            color: #4b5563;
            line-height: 1.7;
            font-size: 15px;
        }}
        
        .result {{
            color: #374151;
            line-height: 1.6;
            font-size: 14px;
        }}
        
        .objects {{
            max-height: 300px;
            overflow-y: auto;
            font-size: 13px;
            line-height: 1.8;
            color: #4b5563;
            font-family: 'Monaco', 'Courier New', monospace;
            white-space: pre-wrap;
        }}
        
        .objects::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .objects::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 4px;
        }}
        
        .objects::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 4px;
        }}
        
        .objects::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Planner Trace Log</h1>
            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-label">Total Steps</div>
                    <div class="stat-value">{total_steps}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Successful</div>
                    <div class="stat-value" style="color: #10b981;">{successes}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-value">{success_rate:.1f}%</div>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="task-section">
                <h2>📋 Task</h2>
                <p>{task}</p>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="toggleAll()">Expand/Collapse All</button>
                <button class="btn filter-btn" onclick="filterSteps('all')">All</button>
                <button class="btn filter-btn" onclick="filterSteps('success')">✓ Success</button>
                <button class="btn filter-btn" onclick="filterSteps('failure')">✗ Failures</button>
            </div>
            
            <div class="steps">
"""

    # Add each step
    for idx, step in enumerate(steps):
        step_num = idx + 1
        thought = step.get('thought', 'No thought')
        action = step.get('action', 'Unknown')
        args = step.get('args', '')
        result = step.get('result', 'No result')
        objects = step.get('objects', '')
        success = step.get('success', False)

        status_class = 'status-success' if success else 'status-failure'
        status_text = '✓ Success' if success else '✗ Failed'
        step_class = 'success' if success else 'failure'

        action_display = f"{action}[{args}]" if args else action

        html_content += f"""
                <div class="step {step_class}" data-status="{step_class}">
                    <div class="step-header" onclick="toggleStep(this.parentElement)">
                        <div class="step-number">{step_num}</div>
                        <div class="step-action">{action_display}</div>
                        <div class="step-status {status_class}">{status_text}</div>
                    </div>
                    <div class="step-content">
                        <div class="section">
                            <div class="section-label">💭 Thought</div>
                            <div class="section-content">
                                <div class="thought">{thought}</div>
                            </div>
                        </div>
                        
                        <div class="section">
                            <div class="section-label">📤 Result</div>
                            <div class="section-content">
                                <div class="result">{result}</div>
                            </div>
                        </div>
"""

        if objects:
            html_content += f"""
                        <div class="section">
                            <div class="section-label">📦 Objects Discovered</div>
                            <div class="section-content">
                                <div class="objects">{objects}</div>
                            </div>
                        </div>
"""

        html_content += """
                    </div>
                </div>
"""

    html_content += """
            </div>
        </div>
    </div>
    
    <script>
        function toggleStep(element) {
            element.classList.toggle('expanded');
        }
        
        let allExpanded = false;
        function toggleAll() {
            const steps = document.querySelectorAll('.step');
            allExpanded = !allExpanded;
            steps.forEach(step => {
                if (allExpanded) {
                    step.classList.add('expanded');
                } else {
                    step.classList.remove('expanded');
                }
            });
        }
        
        function filterSteps(filter) {
            const steps = document.querySelectorAll('.step');
            const buttons = document.querySelectorAll('.filter-btn');
            
            // Update button states
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            steps.forEach(step => {
                if (filter === 'all') {
                    step.style.display = 'block';
                } else {
                    const status = step.getAttribute('data-status');
                    if (status === filter) {
                        step.style.display = 'block';
                    } else {
                        step.style.display = 'none';
                    }
                }
            });
        }
        
        // Auto-expand first step
        document.querySelector('.step').classList.add('expanded');
    </script>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ HTML file generated: {output_file}")
    print(f"  Total steps: {total_steps}")
    print(f"  Success rate: {success_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Convert PARTNR trace logs to HTML')
    parser.add_argument('trace_file', help='Path to the trace text file')
    parser.add_argument('--output', '-o', help='Output HTML file (default: same name as input with .html extension)')
    parser.add_argument('--open', action='store_true', help='Open the HTML file in browser after generation')

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = Path(args.trace_file).with_suffix('.html')

    # Parse and convert
    print(f"Parsing trace file: {args.trace_file}")
    trace_data = parse_trace_file(args.trace_file)

    print(f"Generating HTML...")
    generate_html(trace_data, str(output_file))

    # Open in browser if requested
    if args.open:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        print(f"✓ Opened in browser")


if __name__ == '__main__':
    main()
