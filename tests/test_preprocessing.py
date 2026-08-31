def _make_doc(n_words):
    return " ".join(f"w{i}" for i in range(n_words))


def test_long_doc_last_chunk_is_true_ending():
    from src.preprocessing import smart_split_chunks
    doc = _make_doc(12000)          # > 24 raw chunks -> selection kicks in
    chunks = smart_split_chunks(doc, max_chunks=24)
    assert len(chunks) == 24
    assert chunks[-1].split()[-1] == "w11999"      # last stored chunk holds the ending
    assert chunks[0].split()[0] == "w0"            # first chunk holds the beginning


def test_long_doc_chunks_are_in_document_order():
    from src.preprocessing import smart_split_chunks
    doc = _make_doc(12000)
    chunks = smart_split_chunks(doc, max_chunks=24)
    first_word_indices = [int(c.split()[0][1:]) for c in chunks]
    assert first_word_indices == sorted(first_word_indices)


def test_short_doc_unchanged():
    from src.preprocessing import smart_split_chunks
    doc = _make_doc(1000)           # few chunks -> no selection, natural order
    chunks = smart_split_chunks(doc, max_chunks=24)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w999"


def test_chunks_do_not_overlap():
    from src.preprocessing import smart_split_chunks
    doc = _make_doc(1000)
    chunks = smart_split_chunks(doc, max_chunks=24)
    seen = set()
    for c in chunks:
        words = set(c.split())
        assert words.isdisjoint(seen)     # no word appears in two chunks
        seen |= words
    assert " ".join(chunks) == doc        # chunks partition the document


def test_removing_last_chunk_removes_the_ending():
    """Verdict-leak regression: after dropping the final chunk, none of the
    document's last words may survive in the remaining chunks."""
    from src.preprocessing import smart_split_chunks
    # 730 words: with the old 50-word overlap the penultimate chunk contained
    # the entire short final chunk (the 'verdict').
    doc = _make_doc(730)
    chunks = smart_split_chunks(doc, max_chunks=24)
    remaining = " ".join(chunks[:-1]).split()
    assert "w729" not in remaining
    assert chunks[-1].split()[-1] == "w729"


def test_long_doc_selection_dedupes_identical_chunk_text():
    from src.preprocessing import smart_split_chunks
    # 30 chunks of 400 words: 2 unique start, 23 identical middle, 5 unique end
    words = ([f"s{i}" for i in range(800)]
             + ["mid"] * (400 * 23)
             + [f"e{i}" for i in range(2000)])
    chunks = smart_split_chunks(" ".join(words), max_chunks=24)
    assert len(chunks) == len(set(chunks))   # no duplicate chunk text


def test_split_small_corpus_has_no_empty_split():
    from src.preprocessing import DataController
    ctrl = DataController(train_frac=0.80, val_frac=0.10, test_frac=0.10, seed=1)
    texts = [f"doc{i}" for i in range(9)]
    tr_t, _, va_t, _, te_t, _ = ctrl.split(texts, [i % 2 for i in range(9)])
    assert len(tr_t) > 0 and len(va_t) > 0 and len(te_t) > 0
    assert len(tr_t) + len(va_t) + len(te_t) == 9


def test_split_all_test_still_works():
    from src.preprocessing import DataController
    ctrl = DataController(train_frac=0.0, val_frac=0.0, test_frac=1.0, seed=1)
    tr_t, _, va_t, _, te_t, _ = ctrl.split(["a", "b", "c"], [0, 1, 0])
    assert (len(tr_t), len(va_t), len(te_t)) == (0, 0, 3)


def test_split_too_small_corpus_raises():
    import pytest
    from src.preprocessing import DataController
    ctrl = DataController(train_frac=0.80, val_frac=0.10, test_frac=0.10, seed=1)
    with pytest.raises(ValueError):
        ctrl.split(["only-doc"], [1])


def test_label_extraction_accept_reject_unclear():
    from src.preprocessing import extract_judgment_label
    assert extract_judgment_label("... the appeal allowed.") == 1
    assert extract_judgment_label("... the appeal dismissed.") == 0
    assert extract_judgment_label("the weather is nice today") == -1
