# thesis-ai-infer

배포 방식별 작업 부하 실험용 CPU 전용 소형 이미지 분류 추론 서비스. MobileNetV2(ONNX Runtime), 포트 8080.

- `GET /` → `ai-infer linux/<arch> host=<hostname> model=mobilenetv2-12 uptime=<초>s` (모델을 읽은 뒤에 포트를 연다)
- `GET /healthz` → `ok`
- `GET /infer` → 내장 합성 JPEG(640x480)를 요청마다 해독·전처리·추론해 상위 5개 클래스 JSON 반환
- `POST /infer` → 요청 본문의 이미지(최대 5 MiB)로 같은 처리

Dockerfile은 넣지 않아 플랫폼의 Python 자동 생성 경로(`python:3.12-slim`, `requirements.txt`, `python main.py`)를 쓴다.
`ORT_THREADS`로 ONNX Runtime 연산 스레드 수를 정할 수 있다(기본 0 = 자동). 모델 출처와 라이선스는 `model/NOTICE.md`.
