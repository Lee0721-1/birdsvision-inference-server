# BirdsVision Inference Server

BirdsVision 的开源准备推理服务器。包含 FastAPI 图片识别接口、挑战令牌鉴权、限流、上传边界、双视图 logits 融合，以及按客户端版本选择的 SOYOL Detect + 分类器链路。两端 1.0.1 及无版本号请求保持旧响应；1.0.2 及以上请求可使用手动画框并返回学名。测试使用 fake 模型，不会下载或加载生产模型。

仓库不包含正式分类器权重、正式类表、SOYOL 权重、图片、服务器地址、TLS 私钥或生产环境变量。直接运行 Uvicorn 时默认只监听 `127.0.0.1`；Docker 容器内监听 `0.0.0.0`，才能使用容器端口映射。缺少自行配置的旧分类器权重时服务拒绝启动；缺少新版所需的三个路径时，1.0.2 请求返回模型未就绪，不会冒充旧版识别结果。

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

容器使用同一组私有环境变量，并将自有权重和匹配的类表只读挂载到容器内；镜像与仓库都不包含它们。容器对外暴露 8000 端口，健康检查查询容器内的 `/api/health`。公网部署仍需反向代理、TLS 和访问控制。

公网服务应放在反向代理和 TLS 后面，并由部署方自行设置访问控制、容量限制、日志保留和安全更新。只要服务对公网开放，知道接口地址且满足鉴权规则的人就可能调用它；公开源码本身不会自动开放任何服务器。

## 1.0.2 SOYOL 路由

SOYOL 取自 Student YOLO；内部教师模型 Teacher YOLO 简写为 TYLO。TYLO 是闭源内部模型，主要用于比对学生模型的效果。

`modern_inference.py` 通过三个私有环境变量分别加载匹配的分类器权重、类表和单类 SOYOL Detect 权重；它不提供这些文件。SOYOL 显式使用 one-to-many 分支，`imgsz=640`、`conf=0.25`、`iou=0.7`，经 NMS 后最多 10 框；零框时仅用原图，定位错误直接报错。分类器一次处理原图及裁剪视图，使用随源码写明的融合与分区校准参数。`bird_box` 仅适用于 1.0.2 及以上，且与版本号共同绑定到 HMAC 签名。

这个仓库当前仍为 Private，尚未附带 SOYOL 权重、模型卡、许可署名和独立 `final_test` 验收。内部 1.0.2 服务已运行，不应把本仓库当前提交描述为可供网络用户下载的生产对应源码。正式公开前，应核对实际部署文件、依赖版本和构建提交，公开相应源码与模型发布材料。

## 运行版本和对应源码

`GET /api/source` 在配置 `BIRDSVISION_SOURCE_REPOSITORY_URL` 和完整 40 位 `BIRDSVISION_SOURCE_COMMIT` 后返回源码仓库及固定提交链接。两个值必须一起设置，并且仓库和该提交须允许服务用户直接访问。尚未发布时接口返回 503；上线 AGPL 源码获取入口前不得把它视为合规发布完成。

## 许可证

服务器源代码使用 [AGPL-3.0-only](LICENSE)。`openapi/` 下的接口合同单独使用 Apache-2.0，详见 [.reuse/dep5](.reuse/dep5) 和 [LICENSES](LICENSES)。中文说明见 [LICENSE.zh-CN.md](LICENSE.zh-CN.md)。
