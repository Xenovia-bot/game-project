#!/usr/bin/env python3
"""Sync Kaggle notebook %%writefile cells from canonical project sources."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "training" / "kaggle_visdrone_yolox.ipynb"

EMBEDS = {
    3: ROOT / "tools" / "build_dataset.py",
    4: ROOT / "training" / "visdrone_eval.py",
    5: ROOT / "training" / "exps" / "yolox_nano_visdrone.py",
}


def with_notebook_note(path: Path, body: str) -> str:
    if path.name not in {"build_dataset.py", "yolox_nano_visdrone.py"}:
        return body
    if not body.startswith('"""'):
        return body
    end = body.find('"""', 3)
    if end < 0:
        return body
    end += 3
    prefix, rest = body[:end], body[end:]
    rel = path.relative_to(ROOT).as_posix()
    note = f"\n\nBu dosya, projedeki {rel} ile ayni icerige sahiptir.\n"
    close = prefix.rfind('"""')
    prefix = prefix[:close] + note + prefix[close:]
    return prefix + rest


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    for idx, path in EMBEDS.items():
        body = with_notebook_note(path, path.read_text(encoding="utf-8"))
        cell_src = f"%%writefile {path.name}\n{body}"
        if not cell_src.endswith("\n"):
            cell_src += "\n"
        nb["cells"][idx]["source"] = cell_src.splitlines(keepends=True)
        print(f"updated cell {idx} from {path.relative_to(ROOT)}")
    NB_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("notebook rewritten")


if __name__ == "__main__":
    main()
