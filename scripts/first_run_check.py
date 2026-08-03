"""
First-run health check for Ravana.

Verifies import, providers, optional model presence, and prints next steps.
Does not require a camera. Use --download to prefetch the seamless preset.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/first_run_check.py
  .\\.venv\\Scripts\\python.exe scripts/first_run_check.py --download
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Ravana first-run health check")
    ap.add_argument(
        "--download",
        action="store_true",
        help="Prefetch seamless model preset (HyperSwap + GFPGAN + XSeg)",
    )
    ap.add_argument("--models-dir", default="./models")
    args = ap.parse_args()

    print("=== Ravana first-run check ===")
    try:
        import face_swap
        from face_swap.core.model_manager import ModelManager
        from face_swap.core.providers import resolve_ort_providers
    except Exception as e:
        print(f"FAIL: import face_swap ({e})")
        return 1

    print(f"OK   version {face_swap.__version__}")

    try:
        providers = resolve_ort_providers("auto")
        print(f"OK   ORT providers (auto): {providers}")
    except Exception as e:
        print(f"FAIL: resolve providers ({e})")
        return 1

    mgr = ModelManager(models_dir=args.models_dir)
    if args.download:
        print("Downloading preset 'seamless' ...")
        try:
            mgr.ensure_preset("seamless", show_progress=True)
        except Exception as e:
            print(f"FAIL: download ({e})")
            return 1

    rows = mgr.status()
    needed = {"hyperswap", "gfpgan", "xseg", "inswapper"}
    present = {r["name"] for r in rows if r["present"]}
    missing = sorted(needed - present)
    print("OK   model status:")
    for r in rows:
        if r["name"] not in needed and r["name"] not in ("gpen", "codeformer"):
            continue
        flag = "OK" if r["present"] else "MISS"
        print(f"       [{flag}] {r['name']} {r['version']}")

    if missing:
        print(
            f"WARN missing core weights: {', '.join(missing)}\n"
            "     Run: python -m demos.cli models download --preset seamless"
        )
    else:
        print("OK   seamless core weights present")

    print(
        "\nNext:\n"
        "  python -m demos.cli -s source.jpg -t target.jpg -o out.jpg -q seamless\n"
        "  python -m demos.cli evaluate -s source.jpg -t target.jpg -o out.jpg -q seamless\n"
        "  python -m demos.webcam_demo -s source.jpg --device auto --detect-every 3\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
