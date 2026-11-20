"""Bounded, resumable HTTP range transport for large immutable dataset files."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import shutil
import time


def download_ranges(
    url,
    headers,
    target,
    size,
    expected_sha256,
    workers=8,
    part_bytes=256 * 1024 * 1024,
    report=None,
):
    import requests

    target = Path(target)
    if size <= 0 or workers <= 0 or part_bytes <= 0:
        raise ValueError("Range parameters must be positive.")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("A trusted SHA256 is required for ranged downloads.")
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = target.parent / ".parts" / expected_sha256
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [
        (n, start, min(size, start + part_bytes) - 1)
        for n, start in enumerate(range(0, size, part_bytes))
    ]

    def transfer(item):
        number, start, end = item
        part = parts / f"{number:05d}.part"
        length = end - start + 1
        for attempt in range(5):
            present = part.stat().st_size if part.exists() else 0
            if present > length:
                raise RuntimeError(f"Oversized range cache: {part}")
            if present == length:
                return number, length
            offset = start + present
            request_headers = {
                **headers,
                "Accept-Encoding": "identity",
                "Range": f"bytes={offset}-{end}",
            }
            try:
                with requests.get(
                    url, headers=request_headers, stream=True, timeout=(15, 60)
                ) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        delay = response.headers.get("Retry-After", "")
                        delay = int(delay) if delay.isdigit() else 2 ** (attempt + 1)
                        time.sleep(delay)
                        continue
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"Expected partial response, got HTTP {response.status_code}"
                        )
                    expected_range = f"bytes {offset}-{end}/{size}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Server returned an unexpected byte range.")
                    remaining = end - offset + 1
                    with part.open("ab") as stream:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if len(block) > remaining:
                                raise RuntimeError(
                                    "Server returned excess range bytes."
                                )
                            stream.write(block)
                            remaining -= len(block)
                    if remaining == 0:
                        return number, length
            except requests.RequestException:
                # Never include redirects or signed URLs in exception messages.
                if attempt == 4:
                    raise RuntimeError(
                        f"Network retries exhausted for range {number}"
                    ) from None
            time.sleep(2**attempt)
        raise RuntimeError(f"Incomplete range {number} after five attempts")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transfer, item) for item in ranges]
        for future in as_completed(futures):
            _, length = future.result()
            completed += length
            if report is not None:
                report(completed, size)

    assembled = target.with_suffix(target.suffix + ".assembling")
    digest = hashlib.sha256()
    with assembled.open("wb") as output:
        for number, _, _ in ranges:
            with (parts / f"{number:05d}.part").open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
    if assembled.stat().st_size != size or digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            "Assembled file failed SHA256 validation; range cache retained for inspection."
        )
    assembled.replace(target)
    shutil.rmtree(parts)
    return str(target)
