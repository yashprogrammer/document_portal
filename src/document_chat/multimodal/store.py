from __future__ import annotations
import base64
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain.storage import LocalFileStore


class PayloadStore:
    """Thin wrapper around LocalFileStore for multimodal payloads.

    Stores raw payloads keyed by a UUID-like doc_id. Payloads are strings:
    - text/table: raw string content
    - image: file path string to the stored image on disk (base64 can be produced on demand)
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.store = LocalFileStore(str(self.root_dir))

    def put_many(self, items: List[Tuple[str, str]]) -> None:
        """Bulk insert: list of (doc_id, payload). Encodes to bytes."""
        encoded = []
        for k, v in items:
            if isinstance(v, bytes):
                encoded.append((k, v))
            else:
                encoded.append((k, str(v).encode("utf-8")))
        self.store.mset(encoded)

    def get(self, doc_id: str) -> str:
        val = self.store.mget([doc_id])
        if not val:
            return ""
        data = val[0]
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8")
            except Exception:
                return ""
        return str(data)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file into a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


