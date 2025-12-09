# Episode Editor

A visual GUI tool for editing Habitat partnr-planner episode files. Add, view, and manage objects in a scene using an interactive top-down map.

## Features

### Map Visualization

- **Top-down map** - Renders the actual scene from the episode using Habitat simulator
- **Room overlays** - Semi-transparent colored regions showing room boundaries
- **Object markers** - Color-coded circles showing object positions with category letters

### Object Management

- **View objects** - Left sidebar lists all objects in the scene with their details
- **Add objects** - Right sidebar panel for adding new objects to the scene
- **Object search** - Search objects by name, ID, or category from the HSSD database
- **Thumbnails** - Shows object preview images when available (falls back to category emoji)
- **Auto room detection** - Clicking on the map automatically detects the closest room

### Navigation

- **Zoom** - Mouse wheel or +/− buttons to zoom in/out
- **Pan** - Shift+drag or middle mouse button to pan
- **Reset** - ⟲ button to reset to original view
- **Coordinates** - Live coordinate display as you move the mouse

### Data Persistence

- **Save** - Save changes back to the episode JSON file
- **Proper ordering** - Maintains correct object ordering for metadata extraction

## Usage

### Starting the Editor

```bash
# Activate habitat environment
conda activate habitat

# Run the editor with an episode file
python scripts/episode_editor/app.py --episode path/to/episode.json
```

The editor will start at `http://localhost:5000`

### Adding an Object

1. **Search** - Type in the search box or filter by category
2. **Select** - Click an object from the results to select it
3. **Position** - Click on the map to set X/Z coordinates (Y is height, default 0.8)
4. **Room/Receptacle** - Auto-filled based on click position, or select manually
5. **Add** - Click "Add Object" button

### Saving Changes

Click the "Save Episode" button in the left sidebar to save changes to the JSON file.

## File Structure

```
scripts/episode_editor/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Frontend HTML/CSS/JS
└── static/
    └── maps/              # Generated map images
```

## API Endpoints

| Endpoint                 | Method | Description                   |
| ------------------------ | ------ | ----------------------------- |
| `/`                    | GET    | Main editor page              |
| `/api/episode`         | GET    | Get episode data and metadata |
| `/api/map`             | GET    | Get top-down map image        |
| `/api/map/calibration` | GET    | Get map calibration data      |
| `/api/objects`         | GET    | Search object database        |
| `/api/categories`      | GET    | Get available categories      |
| `/api/receptacles`     | GET    | Get receptacles for a room    |
| `/api/thumbnail/<id>`  | GET    | Get object thumbnail image    |
| `/api/add_object`      | POST   | Add new object to episode     |
| `/api/save`            | POST   | Save episode to file          |

## Keyboard Shortcuts

| Action     | Shortcut                         |
| ---------- | -------------------------------- |
| Zoom in    | Scroll up / Click +              |
| Zoom out   | Scroll down / Click −           |
| Pan        | Shift + Drag / Middle mouse drag |
| Reset view | Click ⟲                         |

## Requirements

- Python 3.8+
- Flask
- Habitat-sim (for map generation)
- HSSD dataset (for object database)

## Improvements Needed

- [ ] **Proper room labels** - Room names from semantic scene are not always accurate
- [ ] Move existing object positions
- [ ] Export scene visualization
- [ ] Better receptacle suggestions based on object type

## Related Documentation

- [Adding Objects to Episode (Manual)](../../docs/ADDING_OBJECTS_TO_EPISODE.md) - Manual JSON editing guide
