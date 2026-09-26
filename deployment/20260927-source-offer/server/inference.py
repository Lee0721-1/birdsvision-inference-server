import io
import time
import warnings

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from PIL import Image

import config
import labels as labels_mod


class StableCUBModel(nn.Module):
    """与训练时完全一致的 ConvNeXt-Tiny 结构。"""

    def __init__(self, num_classes: int = 755):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


_model = None
_transform = None
_device = None


def decode_image(image_bytes: bytes) -> Image.Image:
    """完整校验图片并转为 RGB；异常或超大像素图片统一拒绝。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.verify()

            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
                image.load()
                return image
    except Exception as exc:
        raise ValueError("INVALID_IMAGE") from exc


def init_model():
    """启动时调用一次：加载权重 + 构建 transform + 预热。"""
    global _model, _transform, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = StableCUBModel(num_classes=config.NUM_CLASSES)

    state = torch.load(config.MODEL_PATH, map_location=_device)
    _model.load_state_dict(state)
    _model.to(_device).eval()

    data_cfg = timm.data.resolve_model_data_config(_model)
    _transform = timm.data.create_transform(**data_cfg, is_training=False)

    dummy = torch.zeros(1, 3, 224, 224).to(_device)
    with torch.no_grad():
        _model(dummy)
    print(f"[OK] 模型已加载到 {_device}，类别数 {config.NUM_CLASSES}")


def is_ready() -> bool:
    return _model is not None


def predict(image_bytes: bytes, top_k: int = 3):
    """返回 (results列表, elapsed_ms)；图片无法解析时抛 ValueError。"""
    if _model is None or _transform is None or _device is None:
        raise RuntimeError("MODEL_NOT_READY")

    image = decode_image(image_bytes)

    t0 = time.time()
    x = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        out = _model(x)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        k = max(1, min(int(top_k), 10))
        top_p, top_i = torch.topk(probs, k)
    elapsed_ms = int((time.time() - t0) * 1000)

    results = []
    for rank, (p, idx) in enumerate(zip(top_p.tolist(), top_i.tolist()), start=1):
        lab = labels_mod.get_label(idx)
        results.append({
            "rank": rank,
            "class_id": idx,
            "class_key": lab["class_key"],
            "chinese_name": lab["chinese_name"],
            "english_name": lab["english_name"],
            "folk_name": lab.get("folk_name", ""),
            "confidence": round(float(p), 4),
        })
        if lab.get("scientific_names"):
            results[-1]["scientific_names"] = lab["scientific_names"]
    return results, elapsed_ms
