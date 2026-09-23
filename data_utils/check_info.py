import sys
from datasets import load_dataset_builder

sys.stdout.reconfigure(encoding='utf-8')

for lang in ['marathi', 'konkani']:
    builder = load_dataset_builder('ai4bharat/Shrutilipi', lang)
    print(f"\n--- {lang.upper()} ---")
    print("Splits:", builder.info.splits)
    print("Features:", builder.info.features)
    print("Dataset size:", builder.info.dataset_size)
    print("Download size:", builder.info.download_size)
