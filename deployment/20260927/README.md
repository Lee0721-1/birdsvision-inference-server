# 2026-09-27 线上 Python 源码快照

`server/` 中的 Python 文件和 `requirements.txt` 逐文件取自当日运行中的 `/home/ubuntu/birdsvision-server-releases/1.0.2-soyol-split-20260927/server/`。`systemd/` 是当日生效的独立 SOYOL 定位服务单元及 API 覆盖配置；`ops/` 是当日安装的定时运维检查脚本。[source_manifest.json](source_manifest.json)绑定这 15 个与服务器逐文件核对过的文件。它们记录了实际运行的源码与启动关系。仓库根目录的 `birdsvision_server/` 是整理后的公开参考结构，不能用它替代此快照来声称字节一致。

线上 API 由 `server/app.py` 启动；1.0.2 自动框通过 `server/modern_inference.py` 和 `server/locator_client.py` 调用仅监听 `127.0.0.1:8001` 的 `server/birdsvision_locator/app.py`。无版本号及 1.0.1 请求仍走旧分类链路。[model_binding.json](model_binding.json)记录已核对的可考证 A 层重训权重绑定；模型文件、分类器权重、类表、图片和私有环境变量不在本快照中。快照内的 systemd 文件含当日服务器路径，复现时需按自己的目录和权重路径配置，不能直接照搬到其他主机。

`server/requirements.txt` 是部署目录原有的直接依赖清单；[runtime_environment.json](runtime_environment.json)记录系统和 Python 版本，[runtime_versions.txt](runtime_versions.txt)记主要已安装版本，[runtime-freeze.txt](runtime-freeze.txt)记当日虚拟环境的 42 个包版本。Ultralytics 8.4.126 来自部署目录的 `vendor/`；[vendor_provenance.json](vendor_provenance.json)记录从 PyPI 获取的精确 wheel 与线上 361 个包文件逐文件一致的结果，[verify_vendor_record.py](verify_vendor_record.py)可重做比较。CPU PyTorch 轮子仍需匹配的安装源；完整可复现安装及权重发布说明还需补齐。这些文件校验不构成许可证结论。此快照仅证明上述 Python 文件与生效的服务配置，不表示模型权重或训练图片已获发布许可。仓库仍为私密，公网 `/api/source` 尚未提供可访问的源码提交。
