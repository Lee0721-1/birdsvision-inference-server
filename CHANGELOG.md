# Changelog

## Unreleased

- 将活动 SOYOL 定位服务源码及独立依赖清单迁往 `birdsvision-soyol-locator`；分类 API 只保留外部 HTTP 客户端与版本分流。历史生产快照不改动，仓库继续保持私密。

## Unreleased

- 将 SOYOL Detect 移至单独的本机进程；分类服务仅通过固定的本机接口取得鸟框，不加载 Ultralytics 或定位权重。
- 容器监听 `0.0.0.0:8000` 并增加容器内健康检查；本机 Uvicorn 默认监听地址不变。
- 加入 1.0.2 版本分流、手动画框签名绑定、SOYOL Detect 推理链路与可配置的固定源码提交查询入口；不包含任何生产权重或类表。

## 0.1.0 - 2026-09-10

- 第一阶段本地公开快照：参考 API、鉴权、限流、双视图融合与 fake-based 测试。
