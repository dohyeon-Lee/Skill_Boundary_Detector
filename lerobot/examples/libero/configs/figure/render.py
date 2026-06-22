#!/usr/bin/env python3
"""Render an HTML figure to PNG via headless chromium (Playwright).

Usage:  {project_root}/.venv/bin/python render.py <input.html> [out.png] [--sel .fig] [--scale 2]
Screenshots the element matched by --sel (default ``.fig`` → tight crop), else the full page.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--sel", default=".fig", help="CSS selector to crop to (empty → full page)")
    ap.add_argument("--scale", type=float, default=2.0, help="device scale factor (crispness)")
    ap.add_argument("--width", type=int, default=1400)
    args = ap.parse_args()

    src = Path(args.html).resolve()
    out = Path(args.out) if args.out else src.with_suffix(".png")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": 1000},
                                device_scale_factor=args.scale)
        page.goto(src.as_uri())
        page.wait_for_timeout(150)  # let fonts/layout settle
        target = page.query_selector(args.sel) if args.sel else None
        if target is not None:
            target.screenshot(path=str(out))
        else:
            page.screenshot(path=str(out), full_page=True)
        browser.close()
    print(f"rendered → {out}")


if __name__ == "__main__":
    main()
