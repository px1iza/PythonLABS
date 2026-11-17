from findex.tokenize import tokenize


def test_basic_tokenization():
    assert list(tokenize("Hello WORLD Python")) == [
        "hello", "world", "python"
    ]


def test_unicode_normalization():
    assert list(tokenize("CAFÉ cafe\u0301")) == [
        "café", "café"
    ]


def test_apostrophes():
    assert list(tokenize("don't l’amour")) == [
        "don't", "l’amour"
    ]


def test_hyphens():
    assert list(tokenize("well-known state-of-the-art")) == [
        "well-known", "state-of-the-art"
    ]


def test_numbers():
    assert list(tokenize("Python3 2026 123")) == [
        "python3", "2026", "123"
    ]


def test_punctuation():
    assert list(tokenize("Hello, world! test_case.")) == [
        "hello", "world", "test", "case"
    ]


def test_generator():
    result = tokenize("one two three")
    assert iter(result) is result
    assert next(result) == "one"
    assert list(result) == ["two", "three"]
