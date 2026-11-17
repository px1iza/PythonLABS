from findex.corpus import iter_documents


def test_iter_documents(tmp_path):
    (tmp_path / "first.txt").write_text("Hello", encoding="utf-8")
    (tmp_path / "second.txt").write_text("World", encoding="utf-8")

    documents = iter_documents(tmp_path)

    assert iter(documents) is documents
    assert list(documents) == [
        ("first.txt", "Hello"),
        ("second.txt", "World"),
    ]


def test_nested_documents(tmp_path):
    folder = tmp_path / "nested"
    folder.mkdir()
    (folder / "document.txt").write_text("Text", encoding="utf-8")

    assert list(iter_documents(tmp_path)) == [
        ("document.txt", "Text")
    ]


def test_empty_directory(tmp_path):
    assert list(iter_documents(tmp_path)) == []
