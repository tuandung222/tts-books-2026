"""Giao diện dòng lệnh: ``python -m vn_frontend <lệnh> [văn bản]``.

Ví dụ::

    python -m vn_frontend normalize "Ngày 2/9/1945, lúc 14h30"
    python -m vn_frontend g2p "Xin chào Việt Nam" --dialect south
    echo "Giá 50.000đ/kg." | python -m vn_frontend ids
"""
from __future__ import annotations

import argparse
import sys

from . import g2p, normalize_text, parse_syllable, split_sentences, text_to_sequence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vn_frontend", description="Front-end tiếng Việt cho TTS")
    ap.add_argument("command", choices=["normalize", "sentences", "parse", "g2p", "ids"])
    ap.add_argument("text", nargs="?", help="văn bản (bỏ trống để đọc từ stdin)")
    ap.add_argument("--dialect", default="north", choices=["north", "south"])
    args = ap.parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()

    if args.command == "normalize":
        print(normalize_text(text))
    elif args.command == "sentences":
        for s in split_sentences(normalize_text(text)):
            print(s)
    elif args.command == "parse":
        for word in text.split():
            print(word, parse_syllable(word))
    elif args.command == "g2p":
        print(" ".join("|" if t == " " else t for t in g2p(text, args.dialect)))
    else:
        print(" ".join(map(str, text_to_sequence(text, args.dialect))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
