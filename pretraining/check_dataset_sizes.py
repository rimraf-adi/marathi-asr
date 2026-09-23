import os, sys
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HOME"] = r"D:\huggingface_cache"
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

repos = [
    ("ai4bharat/Shrutilipi", ["marathi", "konkani"], ""),
    ("ARTPARK-IISc/Vaani", ["Marathi", "Konkani", "Malvani", "Khandeshi", "Powari", "Lambani"], "audio/"),
]

total = 0
for repo, configs, prefix in repos:
    info = api.dataset_info(repo, files_metadata=True, token=os.getenv("HF_TOKEN"))
    sib = info.siblings
    print(f"\n=== {repo} (latest: {info.sha}) ===")
    for cfg in configs:
        want = f"{prefix}{cfg}/"
        pats = []
        for f in sib:
            p = f.rfilename
            if p.startswith(want) and p.endswith(".parquet"):
                pats.append((p, f.size or 0))
        n = len(pats)
        sz = sum(s for _, s in pats)
        total += sz
        print(f"  {cfg}: {n} shards, {sz/1e9:.2f} GB")
        for p, s in sorted(pats)[:3]:
            print(f"      {p}  {s/1e6:.1f} MB")
        if n == 0:
            samples = [f.rfilename for f in sib if f.rfilename.endswith(".parquet")][:6]
            print(f"      no match for prefix '{want}'; sample parquet paths: {samples}")

print(f"\nTOTAL parquet to cache: {total/1e9:.2f} GB")