# BirdsVision Inference Server

BirdsVision 的开源准备推理服务器。包含 FastAPI 图片识别接口、挑战令牌鉴权、限流、上传边界、双视图 logits 融合，以及按客户端版本选择的识别链路。SOYOL Detect 在独立进程运行，分类服务通过本机 HTTP 接口取得鸟框。两端 1.0.1 及无版本号请求保持旧响应；1.0.2 及以上请求可使用手动画框并返回学名。测试使用 fake 模型，不会下载或加载生产模型。

仓库不包含正式分类器权重、正式类表、SOYOL 权重、图片、服务器地址、TLS 私钥或生产环境变量。分类器的生产权重保持私有。直接运行 Uvicorn 时默认只监听 `127.0.0.1`；Docker 容器内监听 `0.0.0.0`，才能使用容器端口映射。缺少自行配置的旧分类器权重时服务拒绝启动；新版三个配置全部未设时，1.0.2 请求返回模型未就绪；只设置一部分则启动报错，避免以不完整配置提供服务。

## 项目官网

[鸟视 BirdsVision 官网](https://www.birdsvision.com.cn/)介绍 App、当前模型进度、隐私说明与下载方式。本仓库保存推理服务的开源准备代码；官网页面不是本仓库运行实例的对应源码证明。

## 目录

- `birdsvision_server/api/`：接口、鉴权和客户端版本分流；`tests/api/`：对应测试。
- `birdsvision_server/convnext/`：旧版分类推理和双视图分类运算；`tests/convnext/`：对应测试。
- `birdsvision_server/soyol/`：1.0.2 分类编排与定位服务客户端；`birdsvision_locator/`：独立的 SOYOL 定位服务；`tests/soyol/`：版本分流和进程接口测试。
- `birdsvision_server/config.py`：环境变量配置；`openapi/`：接口合同。

## 安装和测试

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

定位服务使用独立 Python 环境，安装 `requirements-locator.txt`。分类服务的 `requirements.txt` 不安装 Ultralytics。

## 使用自有权重启动

复制 `.env.example` 的值到你自己的进程管理器或私有环境文件，设置与权重严格匹配的类表和类别数：

```bash
export BIRDSVISION_MODEL_PATH=/private/path/model.pth
export BIRDSVISION_LABELS_PATH=/private/path/labels.json
export BIRDSVISION_NUM_CLASSES=1224
uvicorn birdsvision_server.api.app:app --host 127.0.0.1 --port 8000
```

容器使用同一组私有环境变量，并将自有权重和匹配的类表只读挂载到容器内；镜像与仓库都不包含它们。容器对外暴露 8000 端口，健康检查查询容器内的 `/api/health`。公网部署仍需反向代理、TLS 和访问控制。

公网服务应放在反向代理和 TLS 后面，并由部署方自行设置访问控制、容量限制、日志保留和安全更新。只要服务对公网开放，知道接口地址且满足鉴权规则的人就可能调用它；公开源码本身不会自动开放任何服务器。

## 1.0.2 SOYOL 路由

SOYOL 取自 Student YOLO；内部教师模型 Teacher YOLO 简写为 TYLO。TYLO 是闭源内部模型，主要用于比对学生模型的效果。

在独立环境中设置 `BIRDSVISION_SOYOL_MODEL_PATH`，启动 `uvicorn birdsvision_locator.app:app --host 127.0.0.1 --port 8001`。定位服务只加载单类 SOYOL Detect 权重，不读取分类器权重或类表。它显式使用 one-to-many 分支，`imgsz=640`、`conf=0.25`、`iou=0.7`，经 NMS 后最多 10 框。

分类服务另外设置 `BIRDSVISION_1983_MODEL_PATH`、`BIRDSVISION_1983_LABELS_PATH` 和 `BIRDSVISION_SOYOL_LOCATOR_URL=http://127.0.0.1:8001/v1/locate`。分类服务仅接受本机定位地址，并核对返回的图片尺寸和鸟框。零框时仅用原图，定位服务出错时识别请求失败，不回退旧模型。分类器一次处理原图及裁剪视图，使用随源码写明的融合与分区校准参数。`bird_box` 仅适用于 1.0.2 及以上，传入后不请求定位服务，且与版本号共同绑定到 HMAC 签名。

两个进程须共享本机网络命名空间；如使用容器，需另行配置使分类容器内的 `127.0.0.1:8001` 指向定位进程。现有单容器 `Dockerfile` 只启动分类服务，不能单独提供新版自动定位链路。不得将定位端口公开到公网。

这个仓库当前仍为 Private。2026-09-27 线上 1.0.2 已切到独立 SOYOL 定位进程；[当日运行源码快照](deployment/20260927/README.md)保存了实际运行的 Python 文件、服务配置和运维检查脚本。仓库根目录的整理版代码与线上文件在包结构及部分实现上有差异，不能将整理版单独描述为线上运行实例的精确对应源码。正式公开前仍须补齐依赖来源、固定发布版本、SOYOL 权重及模型材料，并核对许可边界。分类器的现用权重不在公开范围内。

## 运行版本和对应源码

`GET /api/source` 在配置 `BIRDSVISION_SOURCE_REPOSITORY_URL` 和完整 40 位 `BIRDSVISION_SOURCE_COMMIT` 后返回源码仓库及固定提交链接。两个值必须一起设置，并且仓库和该提交须允许服务用户直接访问。尚未发布时接口返回 503；上线 AGPL 源码获取入口前不得把它视为合规发布完成。

## 许可证

服务器源代码使用 [AGPL-3.0-only](LICENSE)。`openapi/` 下的接口合同单独使用 Apache-2.0，详见 [.reuse/dep5](.reuse/dep5) 和 [LICENSES](LICENSES)。中文说明见 [LICENSE.zh-CN.md](LICENSE.zh-CN.md)。
