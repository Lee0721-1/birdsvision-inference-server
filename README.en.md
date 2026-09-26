# BirdsVision Inference Server

SOYOL stands for Student YOLO. The internal teacher model is called TYLO (Teacher YOLO); it remains closed source and is mainly used to compare the student's results.

Private open-source preparation of the BirdsVision FastAPI contract, authentication, request limits, dual-view fusion, and versioned 1.0.2 route. SOYOL Detect runs in a separate process; the classifier receives boxes over a loopback HTTP interface. Production classifier weights remain private. Weights, labels, private configuration, and datasets are excluded. This repository is not yet a publicly accessible copy of the running production revision.

Server source is AGPL-3.0-only. The OpenAPI document is Apache-2.0.

The [BirdsVision website](https://www.birdsvision.com.cn/) introduces the app, current model progress, privacy information, and download options. This repository contains source prepared for publication; the website alone does not establish that this is the complete corresponding source of the running service.

The `birdsvision_server/api/`, `birdsvision_server/convnext/`, and `birdsvision_server/soyol/` packages contain the HTTP layer, classifier inference, and locator client. `birdsvision_locator/` is the separate SOYOL process. Start the classifier with `uvicorn birdsvision_server.api.app:app --host 127.0.0.1 --port 8000`.

Local Uvicorn runs bind to `127.0.0.1` by default. The Docker image binds to `0.0.0.0:8000` so published container ports work, and checks `/api/health` inside the container. Supply your own authorized model weights and matching labels through private environment variables and read-only mounts; neither is included in the image.

Install `requirements-locator.txt` in a separate Python environment, set `BIRDSVISION_SOYOL_MODEL_PATH`, and run `uvicorn birdsvision_locator.app:app --host 127.0.0.1 --port 8001`. Set the classifier's `BIRDSVISION_1983_MODEL_PATH`, `BIRDSVISION_1983_LABELS_PATH`, and `BIRDSVISION_SOYOL_LOCATOR_URL=http://127.0.0.1:8001/v1/locate`. Both processes must share a loopback network namespace. The locator uses the one-to-many NMS branch with at most ten post-NMS boxes. The current single-container Dockerfile starts only the classifier and does not by itself provide the 1.0.2 automatic locator route. The production service still runs the earlier in-process architecture; this revision has not been deployed. An optional `GET /api/source` endpoint reports a public repository URL and fixed 40-character Git revision only when both source identity variables are configured and the revision is actually accessible.
