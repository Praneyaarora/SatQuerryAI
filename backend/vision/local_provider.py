"""Vision tool backend: the local_model_server sidecar (InternVL-1B or EarthMind-4B,
whichever it was started with — see local_model_server/README.md). The backend does
not need to know which model is loaded; it only talks to one small HTTP contract.
"""

from __future__ import annotations

import base64
from pathlib import Path

import requests

from backend.config.settings import settings


class LocalVisionProvider:
    def interpret(self, image_path: str, query: str) -> dict:
        if not settings.vision_tool_base_url:
            raise ValueError(
                "SATQUERY_VISION_TOOL_PROVIDER=local but SATQUERY_VISION_TOOL_BASE_URL is not set."
            )

        # When the query asks to mark or locate, guide the model to provide box coordinates
        augmented_query = query
        lower_q = query.lower()
        if any(k in lower_q for k in ("mark", "locate", "highlight", "circle", "draw box", "where is")):
            augmented_query = f"{query}\nIf locating a feature, also provide its bounding box as [ymin, xmin, ymax, xmax] or [x_min, y_min, x_max, y_max]."

        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        response = requests.post(
            f"{settings.vision_tool_base_url.rstrip('/')}/infer",
            json={"image_base64": encoded, "query": augmented_query},
            timeout=settings.vision_tool_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = payload.get("text", "")
        
        # Check if model output contains bounding box coordinates
        bbox = payload.get("bbox")
        if not bbox and raw_text:
            bbox = self._extract_bbox_from_text(raw_text)

        return {
            "text": raw_text,
            "bbox": bbox,
            "model": payload.get("model", settings.vision_tool_model),
            "provider": "local",
        }

    @staticmethod
    def _extract_bbox_from_text(text: str) -> list[float] | None:
        import re
        box_match = re.search(
            r"\[\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?\]",
            text,
        )
        if box_match:
            try:
                nums = [float(box_match.group(i)) for i in range(1, 5)]
                if any(n > 1.0 for n in nums):
                    nums = [max(0.0, min(1.0, n / 1000.0)) for n in nums]
                y1, x1, y2, x2 = nums
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)
                if xmax > xmin and ymax > ymin:
                    return [xmin, ymin, xmax, ymax]
            except Exception:
                pass
        return None
