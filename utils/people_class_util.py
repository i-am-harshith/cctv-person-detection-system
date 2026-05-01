def get_class(classes, category_index, boxes, scores, min_score_thresh=0.3):
    """Return person annotations and the person count for one sampled frame."""
    annotations = []
    count = 0

    for index in range(boxes.shape[0]):
        if scores is not None and scores[index] < min_score_thresh:
            continue

        class_id = int(classes[index])
        class_name = category_index.get(class_id, {}).get("name", "N/A")

        # The crowd metric should count only the person class, not all detected
        # objects in the TensorFlow model output.
        if class_name != "person":
            continue

        count += 1
        ymin, xmin, ymax, xmax = boxes[index].tolist()
        annotations.append(
            {
                "class": class_name,
                "score": round(float(scores[index]), 4),
                "bounding_box": {
                    "ymin": float(ymin),
                    "xmin": float(xmin),
                    "ymax": float(ymax),
                    "xmax": float(xmax),
                },
            }
        )

    return annotations, count
