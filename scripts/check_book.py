#!/usr/bin/env python3
"""Kiểm tra chất lượng cấu trúc của giáo trình.

Các kiểm tra:
  * mọi tệp trong mục lục (nav của mkdocs.yml) đều tồn tại;
  * văn bản ở dạng Unicode NFC (tránh lỗi dấu tiếng Việt dạng tổ hợp);
  * khối mã ``` và công thức $$ được đóng/mở cân bằng;
  * mỗi tệp có đúng một tiêu đề cấp 1;
  * các chương/dự án có đủ những mục bắt buộc theo "Quy ước biên soạn";
  * không còn trang khung (đánh dấu <!-- STUB -->), trừ khi dùng --allow-stubs.

Cách dùng:
  python scripts/check_book.py [--allow-stubs] [--stats]
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"

CHAPTER_RE = re.compile(r"^ch\d{2}-.*\.md$")
PROJECT_RE = re.compile(r"^du-an-\d-.*\.md$")

CHAPTER_REQUIRED = [
    (re.compile(r"^# Chương \d+\. ", re.M), "tiêu đề '# Chương N. ...'"),
    (re.compile(r'!!! abstract "Mục tiêu học tập"'), "hộp 'Mục tiêu học tập'"),
    (re.compile(r'!!! info "Kiến thức cần có"'), "hộp 'Kiến thức cần có'"),
    (re.compile(r"^## Lỗi thường gặp và mẹo thực hành\s*$", re.M), "mục 'Lỗi thường gặp và mẹo thực hành'"),
    (re.compile(r"^## Tóm tắt\s*$", re.M), "mục 'Tóm tắt'"),
    (re.compile(r"^## Bài tập\s*$", re.M), "mục 'Bài tập'"),
    (re.compile(r"^## Tài liệu tham khảo\s*$", re.M), "mục 'Tài liệu tham khảo'"),
]

PROJECT_REQUIRED = [
    (re.compile(r"^# Dự án \d+\. ", re.M), "tiêu đề '# Dự án N. ...'"),
    (re.compile(r'!!! abstract "Mục tiêu'), "hộp 'Mục tiêu ...'"),
    (re.compile(r"^## Tài liệu tham khảo\s*$", re.M), "mục 'Tài liệu tham khảo'"),
]

FENCE_RE = re.compile(r"^\s*(```|~~~)")


def nav_files() -> list[str]:
    """Lấy danh sách tệp .md trong nav (đọc thô, không cần PyYAML)."""
    text = MKDOCS.read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1] if "\nnav:" in text else ""
    return re.findall(r"([\w./-]+\.md)\s*$", nav, flags=re.M)


def strip_code(text: str) -> tuple[str, int]:
    """Bỏ nội dung trong khối mã; trả về (văn bản còn lại, số lần mở/đóng fence)."""
    out, fences, inside = [], 0, False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            fences += 1
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return "\n".join(out), fences


def word_count(text: str) -> int:
    body, _ = strip_code(text)
    body = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)
    return len(re.findall(r"\w+", body))


def check_file(path: Path, allow_stubs: bool) -> list[str]:
    errs: list[str] = []
    raw = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)

    if unicodedata.normalize("NFC", raw) != raw:
        errs.append(f"{rel}: văn bản không ở dạng Unicode NFC")

    if "<!-- STUB -->" in raw:
        if not allow_stubs:
            errs.append(f"{rel}: vẫn còn là trang khung (STUB)")
        return errs

    body, fences = strip_code(raw)
    if fences % 2:
        errs.append(f"{rel}: số dấu mở/đóng khối mã ``` không cân bằng")
    if body.count("$$") % 2:
        errs.append(f"{rel}: số ký hiệu $$ không cân bằng")

    h1 = re.findall(r"^# ", body, flags=re.M)
    if len(h1) != 1:
        errs.append(f"{rel}: cần đúng 1 tiêu đề cấp 1, tìm thấy {len(h1)}")

    name = path.name
    required = CHAPTER_REQUIRED if CHAPTER_RE.match(name) else (
        PROJECT_REQUIRED if PROJECT_RE.match(name) else [])
    for pattern, label in required:
        if not pattern.search(raw):
            errs.append(f"{rel}: thiếu {label}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-stubs", action="store_true", help="cho phép trang khung chưa viết")
    ap.add_argument("--stats", action="store_true", help="in số từ của từng chương")
    args = ap.parse_args()

    errors: list[str] = []
    for rel in nav_files():
        if not (DOCS / rel).exists():
            errors.append(f"mkdocs.yml: tệp trong mục lục không tồn tại: docs/{rel}")

    files = sorted(DOCS.rglob("*.md"))
    total = 0
    for f in files:
        errors.extend(check_file(f, args.allow_stubs))
        if args.stats and (CHAPTER_RE.match(f.name) or PROJECT_RE.match(f.name)):
            n = word_count(f.read_text(encoding="utf-8"))
            total += n
            print(f"{n:>7} từ  {f.relative_to(DOCS)}")
    if args.stats:
        print(f"{total:>7} từ  TỔNG (chương + dự án)")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"\n✗ {len(errors)} lỗi", file=sys.stderr)
        return 1
    print(f"✓ {len(files)} tệp Markdown hợp lệ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
