# BirdsVision Inference Server

SOYOL stands for Student YOLO. The internal teacher model is called TYLO (Teacher YOLO); it remains closed source and is mainly used to compare the student's results.

Private open-source preparation of the BirdsVision FastAPI contract, authentication, request limits, dual-view fusion, and versioned 1.0.2 SOYOL Detect route. Production classifier weights, labels, SOYOL weights, private configuration, and datasets are excluded. This repository is not yet a publicly accessible copy of the running production revision.

Server source is AGPL-3.0-only. The OpenAPI document is Apache-2.0.

The [BirdsVision website](https://www.birdsvision.com.cn/) introduces the app, current model progress, privacy information, and download options. This repository contains source prepared for publication; the website alone does not establish that this is the complete corresponding source of the running service.

The `birdsvision_server/api/`, `birdsvision_server/convnext/`, and `birdsvision_server/soyol/` packages separate the HTTP layer, classifier inference, and 1.0.2 locator route. Tests are grouped by the same responsibilities under `tests/`. Start Uvicorn with `birdsvision_server.api.app:app` from the repository root.

Local Uvicorn runs bind to `127.0.0.1` by default. The Docker image binds to `0.0.0.0:8000` so published container ports work, and checks `/api/health` inside the container. Supply your own authorized model weights and matching labels through private environment variables and read-only mounts; neither is included in the image.

For the 1.0.2 route, supply matching classifier weights, 1,983-class labels, and single-class SOYOL Detect weights via the three environment variables shown in `.env.example`. The locator uses the one-to-many NMS branch with at most ten post-NMS boxes. An optional `GET /api/source` endpoint reports a public repository URL and fixed 40-character Git revision only when both source identity variables are configured and the revision is actually accessible. No model weights or final-test acceptance are published here.
