"""
Downloads the Common Voice Marathi subset (1,208 utterances, ~183 MB)
directly from the Vistaar benchmark repository using HTTP Range requests.
Extracts all files into data/vistaar_benchmarks/commonvoice/marathi/
"""

import os
import sys
import io
import zlib
import json
import urllib.request
from pathlib import Path

URL = "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/commonvoice.zip"
TARGET_DIR = Path("data/vistaar_benchmarks/commonvoice/marathi")


def download_commonvoice_marathi():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    wavs_dir = TARGET_DIR / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    print("[CommonVoice] Querying remote archive metadata...")
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        total_size = int(resp.headers["Content-Length"])

    # 1. Fetch Central Directory from the end
    fetch_size = min(total_size, 3 * 1024 * 1024)
    req_cd = urllib.request.Request(URL, headers={"Range": f"bytes={total_size - fetch_size}-{total_size - 1}"})
    with urllib.request.urlopen(req_cd, timeout=30) as resp:
        eocd_chunk = resp.read()

    eocd_pos = eocd_chunk.rfind(b"PK\x05\x06")
    if eocd_pos == -1:
        raise RuntimeError("Failed to locate End of Central Directory signature.")

    cd_size = int.from_bytes(eocd_chunk[eocd_pos + 12:eocd_pos + 16], "little")
    cd_offset = int.from_bytes(eocd_chunk[eocd_pos + 16:eocd_pos + 20], "little")

    req_cd_full = urllib.request.Request(URL, headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"})
    with urllib.request.urlopen(req_cd_full, timeout=30) as resp:
        cd_data = resp.read()

    # Parse entries for Marathi
    p = 0
    mr_entries = []
    while p < len(cd_data):
        if cd_data[p:p + 4] == b"PK\x01\x02":
            method = int.from_bytes(cd_data[p + 10:p + 12], "little")
            comp_size = int.from_bytes(cd_data[p + 20:p + 24], "little")
            uncomp_size = int.from_bytes(cd_data[p + 24:p + 28], "little")
            fn_len = int.from_bytes(cd_data[p + 28:p + 30], "little")
            ex_len = int.from_bytes(cd_data[p + 30:p + 32], "little")
            comm_len = int.from_bytes(cd_data[p + 32:p + 34], "little")
            header_offset = int.from_bytes(cd_data[p + 42:p + 46], "little")
            fn = cd_data[p + 46:p + 46 + fn_len].decode("utf-8", errors="ignore")

            if fn.startswith("commonvoice/marathi/") and not fn.endswith("/"):
                mr_entries.append({
                    "filename": fn,
                    "method": method,
                    "comp_size": comp_size,
                    "uncomp_size": uncomp_size,
                    "offset": header_offset,
                })
            p += 46 + fn_len + ex_len + comm_len
        else:
            break

    print(f"[CommonVoice] Found {len(mr_entries)} Marathi entries in remote archive.")
    if not mr_entries:
        raise RuntimeError("No Marathi files found in commonvoice.zip!")

    mr_entries.sort(key=lambda x: x["offset"])
    start_byte = mr_entries[0]["offset"]
    end_byte = mr_entries[-1]["offset"] + mr_entries[-1]["comp_size"] + 2048
    span_mb = (end_byte - start_byte) / (1024 * 1024)
    print(f"[CommonVoice] Streaming contiguous Marathi block: bytes {start_byte} to {end_byte} ({span_mb:.2f} MB)...")

    # 2. Download the contiguous block in chunks
    req_block = urllib.request.Request(URL, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    block_data = bytearray()
    with urllib.request.urlopen(req_block, timeout=120) as resp:
        downloaded = 0
        chunk_size = 4 * 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            block_data.extend(chunk)
            downloaded += len(chunk)
            sys.stdout.write(f"\r  Downloaded: {downloaded / (1024 * 1024):.1f} / {span_mb:.1f} MB ({(downloaded / (end_byte - start_byte + 1)) * 100:.1f}%)")
            sys.stdout.flush()
    print("\n[CommonVoice] Download complete. Extracting files...")

    # 3. Extract and write each file
    extracted_count = 0
    for entry in mr_entries:
        rel_offset = entry["offset"] - start_byte
        # Local file header check
        if block_data[rel_offset:rel_offset + 4] != b"PK\x03\x04":
            print(f"Warning: Signature mismatch for {entry['filename']}")
            continue

        fn_len = int.from_bytes(block_data[rel_offset + 26:rel_offset + 28], "little")
        ex_len = int.from_bytes(block_data[rel_offset + 28:rel_offset + 30], "little")
        data_start = rel_offset + 30 + fn_len + ex_len
        data_end = data_start + entry["comp_size"]
        raw_data = block_data[data_start:data_end]

        if entry["method"] == 8:
            file_bytes = zlib.decompress(raw_data, -15)
        elif entry["method"] == 0:
            file_bytes = bytes(raw_data)
        else:
            print(f"Unsupported compression method {entry['method']} for {entry['filename']}")
            continue

        # Save to disk
        rel_fn = entry["filename"].replace("commonvoice/marathi/", "")
        dest_path = TARGET_DIR / rel_fn
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f_out:
            f_out.write(file_bytes)
        extracted_count += 1

    print(f"[CommonVoice] Successfully extracted {extracted_count} files into {TARGET_DIR.resolve()}")

    manifest_p = TARGET_DIR / "manifest.json"
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        print(f"[CommonVoice] Verified manifest with {len(lines)} utterances at {manifest_p.resolve()}")


if __name__ == "__main__":
    download_commonvoice_marathi()
