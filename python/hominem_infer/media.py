from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException


ALLOW_REMOTE_MEDIA = os.getenv("INFER_ALLOW_REMOTE_MEDIA", "false").strip().lower() == "true"


def extract_media(messages: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    images: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "input_image":
                images.append(str(item.get("image_url") or ""))
            elif item_type == "image_url":
                image_url = item.get("image_url") or {}
                images.append(str(image_url.get("url") or ""))
    images = [img for img in images if img]
    return images, []


@dataclass(frozen=True)
class VideoSpec:
    ele: Dict[str, Any]
    is_file_like: bool


def normalize_video_elements(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[VideoSpec]]:
    """
    Convert OpenAI-style video elements into the MLX-VLM expected schema:
      - {"type":"video_url","video_url":{"url":...}} -> {"type":"video","video": "..."}
    Returns a deep-ish copy of messages (only content lists are copied) plus detected videos.
    """
    out_messages: List[Dict[str, Any]] = []
    videos: List[VideoSpec] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            out_messages.append(msg)
            continue
        new_content: List[Any] = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            item_type = (item.get("type") or "").strip()
            if item_type in ("video_url", "input_video"):
                video_url = item.get("video_url") or {}
                url = video_url.get("url") if isinstance(video_url, dict) else None
                if url:
                    ele = dict(item)
                    ele.pop("video_url", None)
                    ele["type"] = "video"
                    ele["video"] = str(url)
                    new_content.append(ele)
                    videos.append(VideoSpec(ele=ele, is_file_like=True))
                    continue
            if item_type == "video":
                ele = dict(item)
                new_content.append(ele)
                videos.append(VideoSpec(ele=ele, is_file_like=isinstance(ele.get("video"), str)))
                continue
            new_content.append(item)
        new_msg = dict(msg)
        new_msg["content"] = new_content
        out_messages.append(new_msg)
    return out_messages, videos


def materialize_video_path(path_or_url: str) -> str:
    """
    Ensure OpenCV can read the video (local path/file://, or optionally http(s)/data url).
    Returns a local filesystem path.
    """
    raw = str(path_or_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty video URL/path.")
    if raw.startswith("file://"):
        return raw[7:]
    if raw.startswith("data:video"):
        if "base64," not in raw:
            raise HTTPException(status_code=400, detail="data:video URL must be base64-encoded.")
        header, b64 = raw.split("base64,", 1)
        suffix = ".mp4"
        if ";" in header:
            mime = header.split(";", 1)[0].removeprefix("data:")
            if mime.endswith("/webm"):
                suffix = ".webm"
            elif mime.endswith("/mp4"):
                suffix = ".mp4"
        try:
            import base64

            data = base64.b64decode(b64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 video data: {exc}") from exc
        fd, tmp_path = tempfile.mkstemp(prefix="hominem_video_", suffix=suffix)
        os.close(fd)
        Path(tmp_path).write_bytes(data)
        return tmp_path
    if raw.startswith("http://") or raw.startswith("https://"):
        if not ALLOW_REMOTE_MEDIA:
            raise HTTPException(
                status_code=400,
                detail="Remote video URLs are disabled (set INFER_ALLOW_REMOTE_MEDIA=true to enable).",
            )
        try:
            import requests
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=f"requests not installed: {exc}") from exc
        fd, tmp_path = tempfile.mkstemp(prefix="hominem_video_", suffix=".mp4")
        os.close(fd)
        with requests.get(raw, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return tmp_path
    return raw

