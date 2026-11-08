"""vn_frontend — front-end văn bản tiếng Việt cho TTS (Dự án 1 của giáo trình).

Python thuần, không phụ thuộc thư viện ngoài. API công khai::

    from vn_frontend import (normalize_text, split_sentences, parse_syllable,
                             g2p, text_to_sequence, SYMBOLS)

Thanh điệu được đánh số 1..6 = ngang, huyền, sắc, hỏi, ngã, nặng.
"""
from .g2p import DIALECTS, PHONES, PUNCTUATION, TONE_TOKENS, WORD_BOUNDARY, g2p, syllable_to_phones
from .normalize import normalize_text, read_number
from .numbers import (
    DEFAULT_STYLE, NumberStyle, decimal_to_words, digits_to_words, number_to_words,
    ordinal_to_words, roman_to_int,
)
from .orthography import TONE_NAMES, clean_unicode, normalize_tone_marks, split_tone
from .sentences import split_sentences
from .syllable import Syllable, is_vietnamese_syllable, parse_syllable
from .symbols import (
    BLANK_ID, PAD_ID, SPACE_ID, SYMBOL_TO_ID, SYMBOLS, UNK_ID, intersperse,
    sequence_to_symbols, text_to_sequence,
)

__version__ = "0.1.0"

__all__ = [
    "normalize_text", "split_sentences", "parse_syllable", "g2p", "text_to_sequence", "SYMBOLS",
    "Syllable", "is_vietnamese_syllable", "syllable_to_phones", "number_to_words",
    "digits_to_words", "decimal_to_words", "ordinal_to_words", "roman_to_int", "read_number",
    "NumberStyle", "DEFAULT_STYLE", "clean_unicode", "normalize_tone_marks", "split_tone",
    "TONE_NAMES", "SYMBOL_TO_ID", "PAD_ID", "BLANK_ID", "SPACE_ID", "UNK_ID", "intersperse",
    "sequence_to_symbols", "PHONES", "PUNCTUATION", "TONE_TOKENS", "WORD_BOUNDARY", "DIALECTS",
]
