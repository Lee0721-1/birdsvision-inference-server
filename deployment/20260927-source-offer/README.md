# 2026-09-27 源码入口部署快照

本目录保存当日从 `1.0.2-soyol-split-20260927` 派生、已启动为 `1.0.2-soyol-source-20260927` 的生产版本。`server/app.py` 新增 `GET /api/source`；`server/config.py` 要求源码仓库 URL 与完整提交一起配置。`systemd/30-source-offer.conf` 使分类 API 使用新隔离目录；SOYOL 定位服务仍使用原独立进程。旧发布目录和 `20-soyol-split.conf` 留作回退。

[source_manifest.json](source_manifest.json)记录与运行目录及生效服务配置核对的 16 个源码和配置文件。与[前一快照](../20260927/README.md)相比，API 源码只增加源码入口，配置只增加源码地址验证，其余应用源码一致。根目录 `birdsvision_server/` 仍是重整后的参考实现；这里的 `server/` 才对应此时运行的文件。

2026-09-27 切换后，两项服务均为 active，使用真实公网鉴权回归无版本号、1.0.1、1.0.2 自动框及 1.0.2 手动画框四条路径，均返回 200。公网 `/api/source` 此时返回 503 `SOURCE_NOT_PUBLISHED`，因为三个 GitHub 仓库仍为 Private，未配置固定公开提交。不能把这个已安装但尚未开启的入口视作已经向网络用户提供源码。

`model_binding.json`、运行环境及第三方依赖证据沿用前一快照。模型权重、分类器类表、私有环境文件、原图不在仓库中。实际公开时，需先让对应源码提交与 SOYOL 权重可匿名访问，再配置固定提交，验证 `/api/source` 返回可用链接；还须核对新配置没有改变识别与版本更新行为。
