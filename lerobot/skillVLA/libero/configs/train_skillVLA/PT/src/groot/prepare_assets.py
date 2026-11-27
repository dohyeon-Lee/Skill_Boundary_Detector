#!/usr/bin/env python3
"""Fetch the native processor sidecars, without downloading a second VLM's weights."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / 'models').is_dir())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=root / 'models/groot_n1_7_base')
    args = parser.parse_args()
    config = json.loads((args.base / 'config.json').read_text())
    from huggingface_hub import snapshot_download
    patterns = ['*.json', '*.jinja', '*.txt', '*.model', '*.tiktoken']
    cache = root / '.cache/huggingface/hub'
    try:
        source = Path(snapshot_download(config['model_name'], cache_dir=cache, local_files_only=True))
        required = ['tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json']
        if any(not (source / name).is_file() for name in required):
            raise FileNotFoundError('Cached processor sidecars are incomplete')
    except (OSError, FileNotFoundError):
        source = Path(snapshot_download(config['model_name'], cache_dir=cache,
                                        allow_patterns=patterns, max_workers=4))
    target = args.base / 'processor'
    target.mkdir(exist_ok=True)
    for pattern in patterns:
        for path in source.glob(pattern):
            shutil.copy2(path, target / path.name)
    print(f"GR00T processor assets: {args.base / 'processor'}")


if __name__ == '__main__':
    main()
