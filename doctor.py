from __future__ import annotations

import argparse
import json
import sys

from alignment_engine import _load_runtime, runtime_info


def main() -> int:
    parser = argparse.ArgumentParser(description="LRC Extandator Forced Alignment runtime check")
    parser.add_argument("--preload", action="store_true", help="download/load MMS_FA now")
    args = parser.parse_args()

    info = runtime_info()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["device"] == "unavailable":
        print("PyTorch/torchaudio runtime is unavailable", file=sys.stderr)
        return 2
    if args.preload:
        _load_runtime(lambda pct, msg: print(f"[{pct:3d}%] {msg}"))
        print(json.dumps(runtime_info(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
