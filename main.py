# thesis-ai-infer는 CPU 전용 소형 이미지 분류 추론 서비스다(MobileNetV2, ONNX Runtime).
# 배포 방식별 작업 부하 실험용이며, 모델을 다 읽은 뒤에 포트를 열어 첫 HTTP 200이 곧 추론 준비 완료가 되게 한다.
import io
import json
import os
import platform
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

PORT = 8080
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "mobilenetv2-12.onnx")
ARCH = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
MAX_BODY = 5 * 1024 * 1024


def sample_jpeg():
    """저작권 문제가 없도록 그라디언트와 도형으로 만든 640x480 JPEG. GET /infer는 요청마다 이 바이트를 해독한다."""
    x = np.linspace(0, 255, 640, dtype=np.uint8)
    y = np.linspace(0, 255, 480, dtype=np.uint8)
    rgb = np.stack([np.tile(x, (480, 1)), np.tile(y[:, None], (1, 640)), np.full((480, 640), 96, np.uint8)], axis=2)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.ellipse((180, 120, 460, 400), fill=(200, 40, 40))
    draw.rectangle((40, 300, 200, 460), fill=(30, 160, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def load_session():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))  # 0이면 ONNX Runtime이 코어 수로 정한다.
    return ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])


started_at = time.monotonic()
SESSION = load_session()
INPUT_NAME = SESSION.get_inputs()[0].name
SAMPLE = sample_jpeg()


def preprocess(data):
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((x - MEAN) / STD)[np.newaxis].astype(np.float32)


def classify(data):
    t0 = time.perf_counter()
    logits = SESSION.run(None, {INPUT_NAME: preprocess(data)})[0][0]
    prob = np.exp(logits - logits.max())
    prob /= prob.sum()
    top = np.argsort(prob)[::-1][:5]
    return {"top5": [[int(i), round(float(prob[i]), 4)] for i in top], "ms": round((time.perf_counter() - t0) * 1000, 2)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # 부하 생성기가 연결을 재사용할 수 있게 한다.

    def send(self, code, body, ctype="text/plain; charset=utf-8"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/healthz":
            self.send(200, "ok\n")
        elif self.path == "/infer":
            self.send(200, json.dumps(classify(SAMPLE)), "application/json")
        elif self.path == "/":
            self.send(200, "ai-infer linux/%s host=%s model=mobilenetv2-12 uptime=%ds\n" % (
                ARCH, socket.gethostname(), time.monotonic() - started_at))
        else:
            self.send(404, "not found\n")

    def do_POST(self):
        if self.path != "/infer":
            self.send(404, "not found\n")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self.send(400, "image body required (max 5 MiB)\n")
            return
        try:
            result = classify(self.rfile.read(length))
        except Exception as e:  # 해독할 수 없는 이미지
            self.send(400, "bad image: %s\n" % type(e).__name__)
            return
        self.send(200, json.dumps(result), "application/json")

    def log_message(self, fmt, *args):  # 부하 실험 중 요청마다 로그를 쓰지 않는다.
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print("listening on :%d (model loaded in %.2fs, arch=%s)" % (PORT, time.monotonic() - started_at, ARCH), flush=True)
    server.serve_forever()
