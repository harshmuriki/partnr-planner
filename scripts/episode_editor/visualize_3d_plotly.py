#!/usr/bin/env python3
"""
3D Apartment Visualizer using Plotly

Interactive 3D visualization with hover coordinates and click events.

Usage:
    python visualize_3d_plotly.py [--file spatial_data_101.json]
"""

import argparse
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = quat
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R


def get_box_corners(center, size, rotation=None):
    """Get the 8 corners of a box with optional rotation."""
    # Half extents
    hx, hy, hz = size[0]/2, size[1]/2, size[2]/2

    # Corners in local space (centered at origin)
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],  # Bottom
        [-hx, -hy, hz],  [hx, -hy, hz],  [hx, hy, hz],  [-hx, hy, hz],   # Top
    ])

    # Apply rotation if provided
    if rotation is not None:
        R = quaternion_to_rotation_matrix(rotation)
        corners = corners @ R.T

    # Translate to center
    corners += center

    return corners


def create_box_mesh(center, size, color, name, rotation=None):
    """Create a box mesh for Plotly."""
    corners = get_box_corners(center, size, rotation)

    # Define the 12 triangular faces (2 triangles per box face)
    faces = [
        # Bottom face
        [0, 1, 2], [0, 2, 3],
        # Top face
        [4, 6, 5], [4, 7, 6],
        # Front face
        [0, 5, 1], [0, 4, 5],
        # Back face
        [2, 7, 3], [2, 6, 7],
        # Left face
        [0, 3, 7], [0, 7, 4],
        # Right face
        [1, 5, 6], [1, 6, 2],
    ]

    # Flatten vertices and faces for Plotly
    i, j, k = [], [], []
    for face in faces:
        i.append(face[0])
        j.append(face[1])
        k.append(face[2])

    return {
        'vertices': corners,
        'faces': (i, j, k),
        'color': color,
        'name': name
    }


def get_color_name_and_rgb(index, total=None):
    """Get a color name and RGB value for an index."""
    colors = [
        ("Red", [0.9, 0.1, 0.1]),
        ("Orange", [0.9, 0.5, 0.1]),
        ("Yellow", [0.9, 0.9, 0.1]),
        ("Lime", [0.5, 0.9, 0.1]),
        ("Green", [0.1, 0.9, 0.1]),
        ("Cyan", [0.1, 0.9, 0.9]),
        ("Blue", [0.1, 0.5, 0.9]),
        ("Purple", [0.5, 0.1, 0.9]),
        ("Magenta", [0.9, 0.1, 0.9]),
        ("Pink", [0.9, 0.1, 0.5]),
        ("Brown", [0.6, 0.3, 0.1]),
        ("Teal", [0.1, 0.6, 0.6]),
        ("Navy", [0.1, 0.2, 0.6]),
        ("Maroon", [0.6, 0.1, 0.2]),
        ("Olive", [0.5, 0.5, 0.1]),
        ("Coral", [0.9, 0.5, 0.3]),
        ("Salmon", [0.9, 0.6, 0.5]),
        ("Gold", [0.9, 0.8, 0.1]),
        ("Violet", [0.7, 0.1, 0.9]),
        ("Indigo", [0.3, 0.1, 0.6]),
    ]
    return colors[index % len(colors)]


def rgb_to_plotly(rgb):
    """Convert RGB [0-1] to Plotly RGB string."""
    return f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'


def load_spatial_data(json_path):
    """Load spatial data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_grid(size=50, spacing=1.0, y_level=0.0, color='lightgray'):
    """Create grid lines."""
    lines_x = []
    lines_y = []
    lines_z = []

    half_size = size / 2 * spacing

    # Lines parallel to X axis
    for i in range(-size//2, size//2 + 1):
        z = i * spacing
        lines_x.extend([None, -half_size, half_size])
        lines_y.extend([None, y_level, y_level])
        lines_z.extend([None, z, z])

    # Lines parallel to Z axis
    for i in range(-size//2, size//2 + 1):
        x = i * spacing
        lines_x.extend([None, x, x])
        lines_y.extend([None, y_level, y_level])
        lines_z.extend([None, -half_size, half_size])

    return go.Scatter3d(
        x=lines_x,
        y=lines_y,
        z=lines_z,
        mode='lines',
        line=dict(color=color, width=1),
        name='Grid',
        showlegend=False,
        hoverinfo='skip'
    )


def visualize_apartment_plotly(spatial_data):
    """Create Plotly visualization from spatial data."""

    fig = go.Figure()

    # Track minimum y for grid placement
    min_y = float('inf')

    # Add rooms as wireframe boxes
    print("\n" + "=" * 80)
    print("ROOM COLOR LEGEND:")
    print("=" * 80)

    room_idx = 0
    for room_name, room_data in spatial_data.get('rooms', {}).items():
        if room_data.get('bounds'):
            bounds = room_data['bounds']
            min_corner = np.array(bounds['min'])
            max_corner = np.array(bounds['max'])

            center = (min_corner + max_corner) / 2
            size = max_corner - min_corner

            color_name, rgb = get_color_name_and_rgb(room_idx)
            print(f"  {room_name:30s} -> {color_name}")

            # Track min y
            min_y = min(min_y, min_corner[1])

            # Create wireframe edges
            corners = get_box_corners(center, size)

            # Define edges
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top
                (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical
            ]

            for edge_idx, (start, end) in enumerate(edges):
                fig.add_trace(go.Scatter3d(
                    x=[corners[start][0], corners[end][0]],
                    y=[corners[start][1], corners[end][1]],
                    z=[corners[start][2], corners[end][2]],
                    mode='lines',
                    line=dict(color=rgb_to_plotly(rgb), width=8),
                    name=f'{room_name}' if edge_idx == 0 else None,
                    showlegend=(edge_idx == 0),
                    hovertemplate=f'<b>{room_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
                ))

            room_idx += 1

    print("=" * 80)

    # Add furniture
    print("\n" + "=" * 80)
    print("FURNITURE COLOR LEGEND:")
    print("=" * 80)

    furniture_idx = 0
    for furniture_name, furniture_data in spatial_data.get('receptacles', {}).items():
        position = np.array(furniture_data.get('position', [0, 0, 0]))
        size = np.array(furniture_data.get('size', [0.5, 0.5, 0.5]))
        rotation = furniture_data.get('rotation')

        # Adjust position so object sits on bottom face
        position[1] += size[1] / 2.0
        min_y = min(min_y, position[1] - size[1] / 2.0)

        color_name, rgb = get_color_name_and_rgb(furniture_idx)
        print(f"  {furniture_name:30s} -> {color_name}")

        # Create box mesh
        box = create_box_mesh(position, size, rgb, furniture_name, rotation)

        fig.add_trace(go.Mesh3d(
            x=box['vertices'][:, 0],
            y=box['vertices'][:, 1],
            z=box['vertices'][:, 2],
            i=box['faces'][0],
            j=box['faces'][1],
            k=box['faces'][2],
            color=rgb_to_plotly(rgb),
            opacity=0.8,
            name=furniture_name,
            hovertemplate=f'<b>{furniture_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
        ))

        furniture_idx += 1

    print("=" * 80)

    # Add grid at floor level
    if min_y != float('inf'):
        grid_y = min_y
    else:
        grid_y = 0.0

    print(f"\nGrid placed at y = {grid_y:.3f}")
    fig.add_trace(create_grid(y_level=grid_y))

    # Add coordinate axes
    axis_length = 2.0
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=4),
        name='X-axis', showlegend=False,
        hovertemplate='X-axis<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='green', width=4),
        name='Y-axis', showlegend=False,
        hovertemplate='Y-axis<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='blue', width=4),
        name='Z-axis', showlegend=False,
        hovertemplate='Z-axis<extra></extra>'
    ))

    # Update layout with click event instructions
    fig.update_layout(
        title={
            'text': "3D Apartment Viewer - Hover to see coordinates | Click on any surface to print coordinates",
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters) - Height',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        hovermode='closest',
        showlegend=True,
        width=1600,
        height=900,
        # Add annotation placeholder for clicked point
        annotations=[
            dict(
                text="Click on any surface to see coordinates here",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.02,
                showarrow=False,
                font=dict(size=14, color="red"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=2
            )
        ]
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize apartment in 3D using Plotly')
    parser.add_argument('--file', type=str, default='static/spatial_data_101.json',
                       help='Path to spatial data JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Save to HTML file instead of opening browser')

    args = parser.parse_args()

    # Load spatial data
    json_path = Path(args.file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return

    print(f"Loading spatial data from: {json_path}")
    spatial_data = load_spatial_data(json_path)
    print(f"Scene ID: {spatial_data.get('scene_id', 'Unknown')}")

    # Create visualization
    fig = visualize_apartment_plotly(spatial_data)

    # Add custom JavaScript for click events
    click_js = """
    <script>
    var myPlot = document.getElementsByClassName('plotly-graph-div')[0];
    var clickedPoints = [];
    var pointCounter = 0;
    
    myPlot.on('plotly_click', function(data){
        // Only process the first clicked point, not all points in the mesh
        if(data.points.length > 0){
            pointCounter++;
            var point = data.points[0];  // Get only the first point
            var x = point.x;
            var y = point.y;
            var z = point.z;
            var label = point.data.name || 'Surface';
            
            // Store clicked point
            clickedPoints.push({x: x, y: y, z: z, label: label, id: pointCounter});
            
            // Print to console
            console.log('🎯 CLICKED POINT #' + pointCounter + ':');
            console.log('   Object: ' + label);
            console.log('   X = ' + x.toFixed(3) + ' m');
            console.log('   Y = ' + y.toFixed(3) + ' m (height)');
            console.log('   Z = ' + z.toFixed(3) + ' m');
            console.log('   Full: (' + x.toFixed(3) + ', ' + y.toFixed(3) + ', ' + z.toFixed(3) + ')');
            console.log('');
            
            // Update annotation at bottom of plot
            var annotation = {
                text: '<b>Point #' + pointCounter + ' - ' + label + '</b><br>' +
                      'X: ' + x.toFixed(3) + ' | Y: ' + y.toFixed(3) + ' | Z: ' + z.toFixed(3),
                xref: 'paper',
                yref: 'paper',
                x: 0.5,
                y: 0.02,
                showarrow: false,
                font: {size: 16, color: 'white'},
                bgcolor: 'rgba(255, 0, 0, 0.8)',
                bordercolor: 'darkred',
                borderwidth: 2,
                borderpad: 8
            };
            
            Plotly.relayout(myPlot, {annotations: [annotation]});
        }
    });
    
    // Add keyboard shortcut to print all clicked points
    document.addEventListener('keydown', function(event) {
        if (event.key === 'p' || event.key === 'P') {
            if (clickedPoints.length === 0) {
                console.log('📍 No points clicked yet. Click on surfaces to record coordinates.');
            } else {
                console.log('');
                console.log('📍 ALL CLICKED POINTS (' + clickedPoints.length + ' total):');
                console.log('='.repeat(60));
                for (var i = 0; i < clickedPoints.length; i++) {
                    var pt = clickedPoints[i];
                    console.log('Point #' + pt.id + ' - ' + pt.label + ':');
                    console.log('  (' + pt.x.toFixed(3) + ', ' + pt.y.toFixed(3) + ', ' + pt.z.toFixed(3) + ')');
                }
                console.log('='.repeat(60));
                console.log('');
            }
        }
        // Clear display with 'c' or 'C' key
        if (event.key === 'c' || event.key === 'C') {
            Plotly.relayout(myPlot, {annotations: []});
            console.log('🧹 Display cleared');
        }
    });
    
    console.log('');
    console.log('🎯 Click on any surface to see its 3D coordinates!');
    console.log('💡 Press P to print all clicked points to console');
    console.log('💡 Press C to clear the display');
    console.log('💡 Open Console (F12) to see all coordinate output');
    console.log('');
    </script>
    """

    # Show or save
    if args.output:
        output_path = Path(args.output)

        # Save with custom JavaScript
        html_string = fig.to_html(include_plotlyjs='cdn')
        html_with_js = html_string.replace('</body>', click_js + '</body>')

        with open(output_path, 'w') as f:
            f.write(html_with_js)

        print(f"\nVisualization saved to: {output_path}")
        print("Open this file in any web browser to view.")
        print("\n" + "=" * 80)
        print("INTERACTIVE FEATURES:")
        print("=" * 80)
        print("  ✓ Hover over surfaces: See coordinates in tooltip")
        print("  ✓ Click on surfaces: Coordinates shown at bottom + logged to console")
        print("  ✓ Press 'P' key: Print all clicked points to console")
        print("  ✓ Press 'C' key: Clear display")
        print("  ✓ Browser Console (F12): See all clicked coordinates logged")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("OPENING INTERACTIVE VISUALIZATION IN BROWSER")
        print("=" * 80)
        print("\nFeatures:")
        print("  - Hover over any surface to see exact (X, Y, display + console)")
        print("  - Press 'P' to see all clicked points in console")
        print("  - Press 'C' to clear displayinates (popup + console)")
        print("  - Press 'P' to see all clicked points")
        print("  - Click and drag to rotate")
        print("  - Scroll to zoom")
        print("  - Right-click and drag to pan")
        print("  - Click legend items to show/hide objects")
        print("  - Click camera icon to reset view")
        print("  - Press F12 to open browser console and see all clicked points")
        print("=" * 80 + "\n")

        # For show(), we need to save temp file with JS
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            html_string = fig.to_html(include_plotlyjs='cdn')
            html_with_js = html_string.replace('</body>', click_js + '</body>')
            f.write(html_with_js)
            temp_path = f.name

        import webbrowser
        webbrowser.open('file://' + temp_path)
        print(f"Opened in browser (temp file: {temp_path})")


if __name__ == "__main__":
    main()
