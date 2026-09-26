# BirdsVision Inference Server

> This repository is not yet publicly accessible. The plan is to publish the classifier API and its recorded production source. The SOYOL service has moved to the independent [SOYOL repository](https://github.com/Lee0721-1/birdsvision-soyol-locator), and classifier training source is in a [separate repository](https://github.com/Lee0721-1/birdsvision-model-training). Old commits and `deployment/20260927/` still contain locator files from before the split; review the full history and running-source correspondence before publication.

SOYOL stands for Student YOLO. The internal teacher model is called TYLO (Teacher YOLO); it remains closed source and is mainly used to compare the student's results.

Source for the BirdsVision FastAPI contract, authentication, request limits, dual-view fusion, and versioned 1.0.2 route. SOYOL Detect runs in a separate project and process; the classifier receives boxes over a loopback HTTP interface without loading Ultralytics or locator weights. Production classifier weights remain private. Weights, labels, private configuration, and datasets are excluded.

Server source is AGPL-3.0-only. The OpenAPI document is Apache-2.0.

The [BirdsVision website](https://www.birdsvision.com.cn/) introduces the app, current model progress, privacy information, and download options. This repository contains source prepared for publication; the website alone does not establish that this is the complete corresponding source of the running service.

The `birdsvision_server/api/`, `birdsvision_server/convnext/`, and `birdsvision_server/recognition/` packages contain the HTTP layer, classifier inference, and the client for the external locator. Start the classifier with `uvicorn birdsvision_server.api.app:app --host 127.0.0.1 --port 8000`.

Local Uvicorn runs bind to `127.0.0.1` by default. The Docker image binds to `0.0.0.0:8000` so published container ports work, and checks `/api/health` inside the container. Supply your own authorized model weights and matching labels through private environment variables and read-only mounts; neither is included in the image.

Set the classifier's `BIRDSVISION_1983_MODEL_PATH`, `BIRDSVISION_1983_LABELS_PATH`, and `BIRDSVISION_SOYOL_LOCATOR_URL=http://127.0.0.1:8001/v1/locate`. The external locator's installation and run instructions are in the [SOYOL repository](https://github.com/Lee0721-1/birdsvision-soyol-locator). Both processes must share a loopback network namespace. The current single-container Dockerfile starts only the classifier and does not by itself provide the 1.0.2 automatic locator route. On 2026-09-27 production switched to separate locator and API processes. [The recorded running Python source snapshot](deployment/20260927/README.md) includes files from both processes and will be part of the source publication after review. An optional `GET /api/source` endpoint reports a public repository URL and fixed 40-character Git revision only when both source identity variables are configured and the revision is actually accessible.
