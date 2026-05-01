# Model Files

This project keeps large TensorFlow model files out of Git.

Expected local model path:

```text
models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
```

You can copy the frozen graph from the original project folder:

```text
person_detection_from_cctv_video-master/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
```

Alternative:

```bash
export CCTV_MODEL_PATH="/absolute/path/to/frozen_inference_graph.pb"
streamlit run app.py
```

Do not commit `.pb`, `.tar.gz`, or other large model binaries to GitHub.
