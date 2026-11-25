"""
Handler para RunPod Serverless (alternativa a RunPod Pod HTTP).
Permite ejecutar el mismo pipeline M1+M2+M3+M4 como worker serverless.

Payload esperado:
{
  "input": {
    "mode": "full" | "placa",
    "image_b64": "<base64 JPEG/PNG>",
    "include_annotated": true
  }
}
"""

import base64
import cv2
import numpy as np
import runpod

from pipeline import TollInferencePipeline

pipeline = TollInferencePipeline()


def handler(event):
    inp = event.get("input", {})
    mode = inp.get("mode", "full")
    b64_str = inp.get("image_b64", "")
    include_annotated = bool(inp.get("include_annotated", True))

    if not b64_str:
        return {"error": "Campo 'image_b64' es obligatorio"}

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    raw = base64.b64decode(b64_str)
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Imagen inválida o corrupta"}

    if mode == "placa":
        return pipeline.infer_plate_only(img)
    return pipeline.infer_full(img, include_annotated=include_annotated)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
