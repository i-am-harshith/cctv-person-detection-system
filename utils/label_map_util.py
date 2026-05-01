import re

import tensorflow as tf


def _read_label_map(path):
    """Read a label map with TensorFlow 2 compatible file I/O."""
    with tf.io.gfile.GFile(path, "r") as file_handle:
        return file_handle.read()


def _extract_value(block, key):
    match = re.search(rf"{key}\s*:\s*\"?([^\"\n]+)\"?", block)
    return match.group(1).strip() if match else None


def load_labelmap(path):
    """Load a simple TensorFlow Object Detection API .pbtxt label map."""
    content = _read_label_map(path)
    items = []

    for block in re.findall(r"item\s*\{(.*?)\}", content, flags=re.DOTALL):
        class_id = _extract_value(block, "id")
        if class_id is None:
            continue

        name = _extract_value(block, "name")
        display_name = _extract_value(block, "display_name")
        items.append(
            {
                "id": int(class_id),
                "name": name or display_name or f"class_{class_id}",
                "display_name": display_name,
            }
        )

    if not items:
        raise ValueError(f"No label-map items found in {path}")

    return items


def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    """Convert parsed label-map rows into TensorFlow-style category dicts."""
    categories = []
    seen_ids = set()

    for item in label_map:
        class_id = item["id"]
        if class_id < 1 or class_id > max_num_classes or class_id in seen_ids:
            continue

        label_name = item["name"]
        if use_display_name and item.get("display_name"):
            label_name = item["display_name"]

        categories.append({"id": class_id, "name": label_name})
        seen_ids.add(class_id)

    return categories


def create_category_index(categories):
    """Create a dictionary keyed by numeric class id."""
    return {category["id"]: category for category in categories}


def create_category_index_from_labelmap(label_map_path):
    """Read a .pbtxt label map and return a category index."""
    label_map = load_labelmap(label_map_path)
    max_num_classes = max(item["id"] for item in label_map)
    categories = convert_label_map_to_categories(label_map, max_num_classes)
    return create_category_index(categories)
