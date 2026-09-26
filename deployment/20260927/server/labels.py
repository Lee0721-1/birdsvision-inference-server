import json
from pathlib import Path
import config

_labels = None


def load_labels():
    global _labels
    if _labels is None:
        with open(config.LABELS_PATH, encoding="utf-8") as f:
            _labels = json.load(f)
        assert len(_labels) == config.NUM_CLASSES, (
            f"labels.json 长度 {len(_labels)} != {config.NUM_CLASSES}"
        )
        for i, item in enumerate(_labels):
            assert item.get("class_id") == i, f"labels.json 索引错位: index={i}, item={item}"
        scientific_path = Path(__file__).resolve().parent / f"data/scientific_names_{len(_labels)}.json"
        if scientific_path.exists():
            with scientific_path.open(encoding="utf-8") as stream:
                scientific_rows = json.load(stream)
            if len(scientific_rows) != len(_labels):
                raise ValueError("scientific-name mapping does not match current labels")
            for index, (label, scientific) in enumerate(zip(_labels, scientific_rows)):
                if (scientific["class_id"] != index or scientific["class_key"] != label["class_key"]
                        or not scientific["scientific_names"]):
                    raise ValueError(f"scientific-name mapping mismatch at {index}")
                label["scientific_names"] = scientific["scientific_names"]
    return _labels


def get_label(class_id: int) -> dict:
    """返回 {class_id, class_key, english_name, chinese_name, folk_name}"""
    labels = load_labels()
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return {
        "class_id": class_id,
        "class_key": f"unknown_{class_id}",
        "english_name": f"class_{class_id}",
        "chinese_name": f"未知类别 {class_id}",
        "folk_name": "",
        "scientific_names": [],
    }
