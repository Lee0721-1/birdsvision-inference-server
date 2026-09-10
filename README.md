# BirdsVision Inference Server

BirdsVision 的公开参考推理服务器。第一阶段包含 FastAPI 图片识别接口、挑战令牌鉴权、限流、上传边界、双视图 logits 融合纯函数、OpenAPI 合同和 fake-based 测试。

仓库不包含正式模型权重、正式类表、YOLO 权重、图片、服务器地址、TLS 私钥或生产环境变量。默认配置只监听 `127.0.0.1`，并且因为缺少私有权重而拒绝启动推理；测试通过 fake 后端验证 API，不会下载或加载生产模型。

## 安装和测试

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## 使用自有权重启动

复制 `.env.example` 的值到你自己的进程管理器或私有环境文件，设置与权重严格匹配的类表和类别数：

```bash
export BIRDSVISION_MODEL_PATH=/private/path/model.pth
export BIRDSVISION_LABELS_PATH=/private/path/labels.json
export BIRDSVISION_NUM_CLASSES=1224
uvicorn app:app --host 127.0.0.1 --port 8000
```

公网服务应放在反向代理和 TLS 后面，并由部署方自行设置访问控制、容量限制、日志保留和安全更新。只要服务对公网开放，知道接口地址且满足鉴权规则的人就可能调用它；公开源码本身不会自动开放任何服务器。

## 双视图和定位器边界

`dual_view_inference.py` 负责视图预算、同批次映射和融合。在线 YOLO 定位器属于可替换组件，第一阶段不附带其依赖和权重，也没有在默认服务中启用。接入时应先完成分类器验证、服务器容量测试和对应许可决定。

## 许可证

服务器源代码使用 [AGPL-3.0-only](LICENSE)。`openapi/` 下的接口合同单独使用 Apache-2.0，详见 [.reuse/dep5](.reuse/dep5) 和 [LICENSES](LICENSES)。中文说明见 [LICENSE.zh-CN.md](LICENSE.zh-CN.md)。
