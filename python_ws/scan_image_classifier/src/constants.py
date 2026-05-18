from __future__ import annotations

LABELS = ["rc_car", "duct_tube", "background"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LABEL_ALIASES = {
    "rc_car": "rc_car",
    "car": "rc_car",
    "rc-car": "rc_car",
    "rc car": "rc_car",
    "duct_tube": "duct_tube",
    "duct-tube": "duct_tube",
    "duct tube": "duct_tube",
    "duct": "duct_tube",
    "tube": "duct_tube",
    "background": "background",
    "other": "background",
    "negative": "background",
    "bg": "background",
}

DEFAULT_IMAGE_TOPIC = "/perception/crop/image"

IMPORT_COLUMNS = [
    "import_id",
    "import_name",
    "created_at",
    "source_topic",
    "num_sequences",
    "num_samples",
    "notes",
]

ANNOTATION_COLUMNS = [
    "sample_id",
    "import_id",
    "import_name",
    "sequence_id",
    "bag_path",
    "image_path",
    "stamp_ns",
    "width",
    "height",
    "source_topic",
    "label",
    "label_id",
    "split",
    "reviewed",
    "annotated_at",
    "auto_label",
    "auto_confidence",
    "notes",
]

VERSION_MANIFEST_COLUMNS = [
    "version_name",
    "sample_id",
    "import_id",
    "import_name",
    "sequence_id",
    "bag_path",
    "stamp_ns",
    "width",
    "height",
    "source_topic",
    "label",
    "label_id",
    "split",
    "reviewed",
    "annotated_at",
    "src_image_path",
    "export_image_path",
    "notes",
]
