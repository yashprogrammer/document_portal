from __future__ import annotations
import base64
import io
import os
import re
from pathlib import Path
from typing import Dict, List

from PIL import Image
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.model_loader import ModelLoader


def _looks_like_base64(sb: str) -> bool:
    return re.match(r"^[A-Za-z0-9+/]+[=]{0,2}$", sb or "") is not None


def _is_image_data(b64data: str) -> bool:
    try:
        header = base64.b64decode(b64data)[:8]
        signatures = (
            (b"\xFF\xD8\xFF"),  # jpg
            (b"\x89PNG\r\n\x1a\n"),  # png
            (b"GIF8"),  # gif
        )
        return any(header.startswith(sig) for sig in signatures)
    except Exception:
        return False


def _resize_base64_image(b64: str, size=(1300, 600)) -> str:
    try:
        img_data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_data))
        resized = img.resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format=img.format or "JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return b64


def split_image_text_types(docs: List) -> Dict[str, List[str]]:
    images: List[str] = []
    texts: List[str] = []
    for d in docs:
        # Unpack Document or raw store payload
        s = d.page_content if isinstance(d, Document) else d
        if isinstance(s, bytes):
            try:
                s = s.decode("utf-8")
            except Exception:
                s = ""
        if not isinstance(s, str):
            s = str(s)
        # 1) base64 image
        if _looks_like_base64(s) and _is_image_data(s):
            images.append(_resize_base64_image(s))
            continue
        # 2) local image path
        try:
            p = Path(s)
            if p.exists() and p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                images.append(_resize_base64_image(b64))
                continue
        except Exception:
            pass
        # 3) plain text
        texts.append(s)
    return {"images": images, "texts": texts}


def img_prompt_func(data: Dict) -> List[HumanMessage]:
    formatted_texts = "\n".join(data["context"]["texts"]) if data["context"]["texts"] else ""
    messages = []
    for image in data["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })
    messages.append({
        "type": "text",
        "text": (
            "You are a helpful assistant. Use the images and text to answer the question.\n"
            f"Question: {data['question']}\n\n"
            f"Context:\n{formatted_texts}"
        ),
    })
    return [HumanMessage(content=messages)]


def _make_splitter(supports_images: bool, max_images: int = 5):
    def _split(docs: List):
        out = split_image_text_types(docs)
        # Enforce provider capability and per-request image cap
        if not supports_images:
            # Drop images for non-vision models; keep only texts
            out["images"] = []
        if len(out["images"]) > max_images:
            out["images"] = out["images"][:max_images]
        return out
    return _split


def build_multimodal_chain(retriever, llm, supports_vision: bool = True, max_images: int = 5):
    chain = ({
        "context": retriever | RunnableLambda(_make_splitter(supports_vision, max_images=max_images)),
        "question": RunnablePassthrough(),
    } | RunnableLambda(img_prompt_func) | llm | StrOutputParser())
    return chain


