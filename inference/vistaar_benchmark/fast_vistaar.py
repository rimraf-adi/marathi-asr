"""
Ultra-fast Remote Zip Extractor for Vistaar Benchmarks.
Uses HTTP Range requests to extract ONLY the Marathi subset directly from remote zips
without downloading multi-gigabyte files for all other languages.
"""

import io
import os
import sys
import time
import zlib
import struct
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP Range stream for reading zip central directories remotely."""
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.length = int(resp.headers.get('Content-Length', 0))
        self.pos = 0

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.length + offset
        return self.pos

    def tell(self) -> int: return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.length: return 0
        end = min(self.pos + len(b) - 1, self.length - 1)
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            b[:len(data)] = data
            self.pos += len(data)
            return len(data)


def download_single_member(url: str, info: zipfile.ZipInfo, output_path: Path):
    """Downloads and decompresses a single member in 1 HTTP Range request."""
    if output_path.exists() and output_path.stat().st_size == info.file_size:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Estimate range: 30 bytes fixed + filename + extra + compressed size
    est_range = 30 + len(info.filename.encode('utf-8')) + 120 + info.compress_size
    req = urllib.request.Request(url, headers={'Range': f'bytes={info.header_offset}-{info.header_offset + est_range}'})
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()

    # Parse local file header (first 30 bytes)
    sig, ver, flag, method, t, d, crc, csize, usize, nlen, elen = struct.unpack('<IHHHHHIIIHH', raw[:30])
    data_start = 30 + nlen + elen
    comp_data = raw[data_start:data_start + info.compress_size]

    if info.compress_type == 8:
        decomp = zlib.decompress(comp_data, -15)
    elif info.compress_type == 0:
        decomp = comp_data
    else:
        raise ValueError(f"Unsupported compress_type: {info.compress_type}")

    with open(output_path, "wb") as f:
        f.write(decomp)
    return True


def extract_marathi_benchmark(benchmark_name: str, zip_url: str, base_out_dir: str = "data/vistaar_benchmarks", max_workers: int = 24) -> Path:
    """Extracts all Marathi files for a benchmark directly from remote zip."""
    out_dir = Path(base_out_dir) / benchmark_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Remote Extract] Reading central directory for {benchmark_name} from {zip_url}...")
    t0 = time.time()
    rz = io.BufferedReader(RemoteZipFile(zip_url))
    zf = zipfile.ZipFile(rz)
    
    # Filter Marathi files
    all_members = zf.infolist()
    mr_members = [
        m for m in all_members 
        if ('marathi' in m.filename.lower() or '/mr/' in m.filename.lower()) 
        and not m.filename.endswith('/')
    ]

    print(f"[Remote Extract] Found {len(mr_members)} Marathi files in {time.time()-t0:.2f}s. Starting parallel download ({max_workers} workers)...")

    def _worker(info: zipfile.ZipInfo):
        # We want target relative to out_dir
        # filename is e.g. "kathbath/marathi/wavs/..."
        # We strip the leading benchmark_name if present
        parts = Path(info.filename).parts
        if parts[0].lower() == benchmark_name.lower():
            rel_path = Path(*parts[1:])
        else:
            rel_path = Path(info.filename)
        dest = out_dir / rel_path
        
        for attempt in range(3):
            try:
                download_single_member(zip_url, info, dest)
                return True
            except Exception as e:
                if attempt == 2:
                    print(f"\n[Warning] Failed to download {info.filename}: {e}")
                    return False
                time.sleep(1)

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_worker, mr_members), total=len(mr_members), desc=f"Downloading {benchmark_name} (mr)"))

    successes = sum(1 for r in results if r)
    print(f"[Remote Extract] Extracted {successes}/{len(mr_members)} files in {time.time()-t1:.2f}s to {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    extract_marathi_benchmark(args.benchmark, args.url, max_workers=args.workers)
