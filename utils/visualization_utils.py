import cv2


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    line_thickness=3,
    draw_only_person=True,
):
    """Draw detection boxes on an OpenCV image array.

    The original project used TensorFlow Object Detection API visualization
    helpers. This compact version keeps the same normalized bounding-box logic
    while drawing only person detections for a cleaner CCTV monitoring output.
    """
    height, width = image.shape[:2]

    for index in range(min(max_boxes_to_draw, boxes.shape[0])):
        score = float(scores[index])
        if score < min_score_thresh:
            continue

        class_id = int(classes[index])
        class_name = category_index.get(class_id, {}).get("name", "N/A")
        if draw_only_person and class_name != "person":
            continue

        ymin, xmin, ymax, xmax = boxes[index]
        if use_normalized_coordinates:
            left = int(xmin * width)
            right = int(xmax * width)
            top = int(ymin * height)
            bottom = int(ymax * height)
        else:
            left, right, top, bottom = int(xmin), int(xmax), int(ymin), int(ymax)

        color = (0, 180, 0)
        cv2.rectangle(image, (left, top), (right, bottom), color, line_thickness)

        label = f"{class_name}: {int(score * 100)}%"
        label_y = max(24, top - 8)
        cv2.putText(
            image,
            label,
            (left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return image
