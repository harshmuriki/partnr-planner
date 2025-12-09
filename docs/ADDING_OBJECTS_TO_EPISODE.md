# Adding Objects to an Episode

This guide explains how to manually add new objects to a PARTNR episode JSON file.

## Prerequisites

- Know which object you want to add (e.g., YCB objects like `024_bowl`, `017_orange`)
- Know where you want to place it (which receptacle/furniture and room)
- Have access to the metadata output to find receptacle handles

## Available Object Sources

Objects can be found in these config directories:

- `data/objects/ycb/configs/` - YCB objects (bowls, fruits, tools, etc.)
- `data/objects_ovmm/train_val/ai2thorhab/configs/objects`
- `data/objects_ovmm/train_val/amazon_berkeley/configs`
- `data/objects_ovmm/train_val/google_scanned/configs`
- `data/objects_ovmm/train_val/hssd/configs/objects`

## ⚠️ Important: Object Metadata CSV Files

**Before adding objects to an episode, ensure they are registered in the metadata CSV files.**

The system looks up object categories from these CSV files in `data/hssd-hab/metadata/`:

1. **`object_categories_filtered.csv`** - For dynamic/movable objects (YCB objects, OVMM objects, etc.)
2. **`fpmodels-with-decomposed.csv`** - For static objects/receptacles (furniture)

### Why This Matters

If an object is not found in the metadata CSV files:
- The object will be labeled as `unknown_<index>` instead of its semantic name (e.g., `bowl_0`)
- This affects object recognition and planning capabilities

### How to Add Objects to the Metadata CSV

**For `object_categories_filtered.csv`:**

The CSV has two columns: `id` and `clean_category`

1. Extract the object shortname from the config filename:
   - `024_bowl.object_config.json` → `024_bowl`
   - `017_orange.object_config.json` → `017_orange`

2. Add a row to the CSV:
   ```csv
   id,clean_category
   024_bowl,bowl
   017_orange,orange
   ```

**Example:**
```bash
echo "024_bowl,bowl" >> data/hssd-hab/metadata/object_categories_filtered.csv
echo "017_orange,orange" >> data/hssd-hab/metadata/object_categories_filtered.csv
```

**Note:** The `id` column should match the object shortname (filename without path, extension, or instance suffix). The `clean_category` should be the semantic category name (e.g., `bowl`, `orange`, `book`, `cup`).

## Steps to Add an Object

### 1. Add to `initial_state`

Add an entry defining the object class, region, and furniture:

```json
{
  "number": 1,
  "object_classes": ["bowl"],
  "allowed_regions": ["hallway_0"],
  "furniture_names": ["table_5"]
}
```

**Important:** Place this entry [index] in the correct position - the order matters!

### 2. Add to `rigid_objs`

Add the object with its transformation matrix:

```json
[
  "024_bowl.object_config.json",
  [
    [r00, r01, r02, x],
    [r10, r11, r12, y],
    [r20, r21, r22, z],
    [0.0, 0.0, 0.0, 1.0]
  ]
]
```

**Transformation Matrix:**

- **Rotation**: 3x3 upper-left matrix (use identity `[[1,0,0],[0,1,0],[0,0,1]]` for no rotation)
- **Position**: 4th column values `[x, y, z]`
  - `x`, `z`: horizontal position
  - `y`: height above ground (match other objects on same surface)

**Tip:** Copy the transformation matrix from an existing object on the same receptacle and adjust x/z slightly.

### 3. Add to `name_to_receptacle`

Map the object handle to its receptacle:

```json
"024_bowl_:0000": "RECEPTACLE_HANDLE|receptacle_mesh_RECEPTACLE_HASH.0000"
```

**Format:**

- Object handle: `OBJECT_NAME_:0000`
- Receptacle: `RECEPTACLE_HANDLE|receptacle_mesh_RECEPTACLE_HASH.XXXX`

Get receptacle handles from the metadata output (`recep_to_handle` field).

## ⚠️ Critical: Order Matters!

The `metadata_extractor.py` maps objects by **index position**:

```python
object_to_handle = {objects[i]: object_handles[i] for i in range(len(objects))}
```

This means:

- The **Nth object** in `initial_state` maps to the **Nth key** in `name_to_receptacle`
- If orders don't match, objects will be assigned to wrong receptacles!

### Correct Order Example

| Position | `initial_state` | `name_to_receptacle` key |
| -------- | ----------------- | -------------------------- |
| 1        | candle            | candle_handle              |
| 2        | candle_holder     | candle_holder_handle       |
| 3        | bowl              | bowl_handle                |
| 4        | orange            | orange_handle              |

**Skip entries:** The "common sense" entry in `initial_state` is automatically skipped (has `"name"` field).

## Complete Example: Adding a Bowl

### Before (existing state)

```
initial_state: [candle, candle_holder]
name_to_receptacle: {candle_handle, candle_holder_handle, ...clutter...}
```

### After (with bowl added)

**1. initial_state:**

```json
{
  "number": 1,
  "object_classes": ["bowl"],
  "allowed_regions": ["hallway_0"],
  "furniture_names": ["table_5"]
}
```

**2. rigid_objs:**

```json
[
  "024_bowl.object_config.json",
  [
    [-0.92743, 0.0, -0.374, -10.5],
    [0.0, 1.0, 0.0, 0.88529],
    [0.374, 0.0, -0.92743, -4.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
]
```

**3. name_to_receptacle (INSERT IN CORRECT POSITION):**

```json
"9dfbc1df..._:0000": "table_5_receptacle...",  // candle (pos 1)
"B075HR7LD2_:0000": "table_5_receptacle...",   // candle_holder (pos 2)
"024_bowl_:0000": "table_5_receptacle...",     // bowl (pos 3) ← INSERT HERE
"Box_15_:0000": "washer_dryer_receptacle...",  // clutter (not tracked)
```

## Finding Receptacle Information

Run the metadata extractor to get receptacle handles:

```bash
python dataset_generation/benchmark_generation/metadata_extractor.py \
  --dataset-path data/datasets/partnr_episodes/v0_0/val_mini/edited_episode.json.gz \
  --save-dir metadata_output/
```

The output JSON contains:

- `recep_to_handle`: Maps furniture names to handles
- `recep_to_room`: Maps furniture to rooms
- `recep_to_description`: Human-readable furniture descriptions

## Finding Object Positions

To place an object on a surface:

1. Find an existing object on the same receptacle
2. Copy its y-coordinate (height)
3. Adjust x and z slightly to avoid overlap
4. Use similar rotation matrix

## Clutter vs Tracked Objects

- **Tracked objects**: Listed in `initial_state`, get semantic names like `bowl_0`
- **Clutter objects**: Only in `rigid_objs` and `name_to_receptacle`, not tracked by metadata extractor

Objects after your tracked objects in `name_to_receptacle` become clutter.

## Verification

After adding an object:

1. **Ensure the object is in the metadata CSV** (see section above)
2. Re-run metadata extraction
3. Check `object_to_recep` and `object_to_room` in output
4. Run visualization to confirm placement
5. **Verify object names**: Objects should appear with semantic names (e.g., `bowl_0`, `orange_0`), not `unknown_<index>`

If the object appears in the wrong location, check the order in `name_to_receptacle`.

If the object appears as `unknown_<index>`, add it to `object_categories_filtered.csv` (see section above).
