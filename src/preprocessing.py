"""PDF extraction, judgment labeling, chunking, and dataset discovery/splitting."""

import calendar
import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

# PDF extraction using PyMuPDF
try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except ImportError:
    PDF_BACKEND = None
    print("WARNING: PyMuPDF not found. Install it with: pip install pymupdf")


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        if PDF_BACKEND == "pymupdf":
            import fitz
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
        else:
            raise RuntimeError("PyMuPDF not available.")
    except Exception as e:
        print(f"  [WARN] Could not read {pdf_path}: {e}")
    return text.strip()


def extract_judgment_label(text):
    tail = text[-500:].lower() if len(text) > 500 else text.lower()
    accept_patterns = [
        r"appeal\s+allowed\.", r"appeals\s+allowed\.", r"petition\s+allowed\.",
        r"petitions\s+allowed\.", r"appeal\s+accepted\.", r"appeals\s+accepted\.",
        r"petition\s+accepted\.", r"petitions\s+accepted\.",
        r"appeal\s+partly\s+allowed\.", r"appeals\s+partly\s+allowed\.",
    ]
    reject_patterns = [
        r"appeal\s+dismissed\.", r"appeals\s+dismissed\.", r"petition\s+dismissed\.",
        r"petitions\s+dismissed\.", r"appeal\s+rejected\.", r"appeals\s+rejected\.",
        r"petition\s+rejected\.", r"petitions\s+rejected\.",
        r"appeal\s+disposed\.", r"appeals\s+disposed\.",
        r"petition\s+disposed\.", r"petitions\s+disposed\.",
    ]
    for pattern in accept_patterns:
        if re.search(pattern, tail):
            return 1
    for pattern in reject_patterns:
        if re.search(pattern, tail):
            return 0

    tail_long = text[-3000:].lower() if len(text) > 3000 else text.lower()
    sentences = re.split(r'\.\s+', tail_long)
    sentences = [s for s in sentences if len(s) > 10]
    sentences.reverse()
    for sentence in sentences[:20]:
        has_appeal = bool(re.search(r'\b(appeal|petition|appeals|petitions)\b', sentence))
        has_not = bool(re.search(r'\b(not|no)\b', sentence))
        if not has_appeal:
            continue
        has_accept = bool(re.search(
            r'\b(allowed|accepted|granted|succeeded|succeeds|in favour of|favor of)\b', sentence))
        has_reject = bool(re.search(
            r'\b(dismissed|rejected|disposed|failed|fails|upheld|affirmed|lacks merit|devoid of merit)\b', sentence))
        if has_not and (has_accept or has_reject):
            continue
        if has_accept and not has_reject:
            return 1
        if has_reject and not has_accept:
            return 0
    return -1


class DataController:
    """Central control for dataset filtering and splitting."""

    def __init__(self, max_files=None, year_range=(1950, 2026), months=None,  # 🚀 DGX: year_range was (2015, 2023) — now full dataset
                 train_frac=0.80, val_frac=0.10, test_frac=0.10,
                 seed=42, min_text_length=50):
        self.max_files = max_files
        self.year_range = year_range
        self.months = months
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.min_text_length = min_text_length
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    def discover_pdfs(self, dataset_root):
        all_pdfs = []
        root = Path(dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        for year_dir in sorted(root.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            if self.year_range is not None:
                if year < self.year_range[0] or year > self.year_range[1]:
                    continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                try:
                    month = int(month_dir.name)
                except ValueError:
                    month = self._month_name_to_num(month_dir.name)
                    if month is None:
                        continue
                if self.months is not None and month not in self.months:
                    continue
                for pdf_path in sorted(month_dir.glob("*.pdf")):
                    all_pdfs.append(str(pdf_path))

        random.seed(self.seed)
        random.shuffle(all_pdfs)
        if self.max_files is not None:
            all_pdfs = all_pdfs[:self.max_files]
        print(f"[DataController] Discovered {len(all_pdfs)} PDFs "
              f"(max_files={self.max_files}, year_range={self.year_range}, months={self.months})")
        return all_pdfs

    def split(self, texts, labels):
        combined = list(zip(texts, labels))
        random.seed(self.seed)
        random.shuffle(combined)
        n = len(combined)
        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)
        # int() truncation can produce an EMPTY split on small corpora
        # (e.g. 9 docs -> n_val = int(0.9) = 0). Every non-zero fraction must
        # get at least one document, or training crashes after a full epoch.
        if self.train_frac > 0 and n_train == 0:
            n_train = 1
        if self.val_frac > 0 and n_val == 0:
            n_val = 1
        n_test = n - n_train - n_val
        if n_train + n_val > n or (self.test_frac > 0 and n_test == 0):
            raise ValueError(
                f"Corpus too small to split {n} docs into "
                f"train/val/test = {self.train_frac}/{self.val_frac}/{self.test_frac}.")
        train = combined[:n_train]
        val = combined[n_train:n_train + n_val]
        test = combined[n_train + n_val:]

        def unzip(pairs):
            if not pairs:
                return [], []
            t, l = zip(*pairs)
            return list(t), list(l)

        tr_t, tr_l = unzip(train)
        va_t, va_l = unzip(val)
        te_t, te_l = unzip(test)
        print(f"[DataController] Split -> Train={len(tr_t)}, Val={len(va_t)}, Test={len(te_t)}")
        return tr_t, tr_l, va_t, va_l, te_t, te_l

    @staticmethod
    def _month_name_to_num(name):
        import calendar
        name_lower = name.strip().lower()
        for i in range(1, 13):
            if (name_lower == calendar.month_name[i].lower()
                    or name_lower == calendar.month_abbr[i].lower()
                    or name_lower == str(i)
                    or name_lower == f"{i:02d}"):
                return i
        return None

    def __repr__(self):
        return (f"DataController(max_files={self.max_files}, "
                f"year_range={self.year_range}, months={self.months}, "
                f"train/val/test={self.train_frac}/{self.val_frac}/{self.test_frac}, "
                f"seed={self.seed})")


def smart_split_chunks(text, max_chunks, chunk_size=400):
    """Split text into NON-overlapping chunks; select first 2 + evenly spaced
    middle + last 5 when there are too many — returned in TRUE DOCUMENT ORDER.

    Fixes the legacy bug where chunks were stored as [first 2, last 5, middle],
    which made "remove the last chunk" delete a middle chunk for long documents
    and leave the verdict visible.

    Chunks must NOT overlap: the old 50-word overlap copied each chunk's first
    50 words into the previous chunk, so for short final chunks the penultimate
    chunk still contained the entire verdict after "remove the last chunk" —
    invalidating the verdict-hidden regime.
    """
    words = text.split()
    raw_chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            raw_chunks.append(chunk)
    total = len(raw_chunks)
    if total <= max_chunks:
        return raw_chunks

    start_count = 2
    end_count = 5
    middle_count = max_chunks - start_count - end_count
    middle_start = start_count
    middle_end = total - end_count

    middle = []
    if middle_end > middle_start and middle_count > 0:
        indices = np.linspace(middle_start, middle_end - 1, middle_count, dtype=int)
        seen = set()
        for idx in indices:               # dedupe repeated indices, keep order
            if idx not in seen:
                seen.add(idx)
                middle.append(raw_chunks[idx])

    selected = raw_chunks[:start_count] + middle + raw_chunks[-end_count:]
    # Dedupe identical chunk TEXT (repeated boilerplate/OCR pages) across the
    # whole selection, preserving document order — matches legacy auto.py.
    selected = list(dict.fromkeys(selected))
    return selected[:max_chunks]


def verify_chunk_order():
    """Self-check: the last stored chunk of a long document must contain the
    document's final words, and no other chunk may contain them (overlap leak).
    Raises at import if either verdict-hiding invariant regresses."""
    doc = " ".join(f"w{i}" for i in range(12000))
    chunks = smart_split_chunks(doc, max_chunks=24)
    if chunks[-1].split()[-1] != "w11999":
        raise RuntimeError(
            "smart_split_chunks is NOT returning chunks in document order — "
            "verdict-hidden experiments would be invalid. Refusing to continue."
        )
    # 730 words -> short final chunk; with any chunk overlap the penultimate
    # chunk would still contain the document ending after the last chunk is
    # removed, silently un-hiding the verdict.
    short_tail = smart_split_chunks(" ".join(f"w{i}" for i in range(730)),
                                    max_chunks=24)
    if "w729" in " ".join(short_tail[:-1]).split():
        raise RuntimeError(
            "smart_split_chunks produces overlapping chunks — removing the "
            "last chunk does not hide the verdict. Refusing to continue."
        )


def load_supreme_court_dataset(config, controller, dataset_root=None, pdf_paths=None):
    """Extract, label, and split a corpus. `dataset_root` overrides
    config.dataset_path (so callers never mutate shared config), and
    `pdf_paths` lets callers reuse an already-discovered file list instead
    of walking the PDF tree a second time."""
    if dataset_root is None:
        dataset_root = config.dataset_path
    if dataset_root is None:
        raise ValueError("config.dataset_path is not set.")
    if pdf_paths is None:
        pdf_paths = controller.discover_pdfs(dataset_root)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under {dataset_root}.")

    texts, labels = [], []
    skipped_read, skipped_label, read_errors = 0, 0, 0
    print(f"\nExtracting text from {len(pdf_paths)} PDFs ...")
    print(f"Using PDF backend: {PDF_BACKEND}")
    for path in tqdm(pdf_paths, desc="Reading PDFs", ncols=100):
        text = extract_text_from_pdf(path)
        if len(text) < controller.min_text_length:
            if len(text) == 0:
                read_errors += 1
            else:
                skipped_read += 1
            continue
        label = extract_judgment_label(text)
        if label == -1:
            skipped_label += 1
            continue
        texts.append(text)
        labels.append(label)

    total_attempted = len(pdf_paths)
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"PDF files attempted: {total_attempted}")
    print(f"Read errors (empty/failed): {read_errors}")
    print(f"Too short (< {controller.min_text_length} chars): {skipped_read}")
    print(f"No clear judgment (label = -1): {skipped_label}")
    print(f"Successfully processed: {len(texts)}")
    print(f"Success rate: {len(texts)/total_attempted*100:.1f}%")
    print(f"\nUsable documents : {len(texts)}")
    print(f"Label distribution   : Accept={labels.count(1)}, Reject={labels.count(0)}")
    if len(texts) == 0:
        raise ValueError("No usable documents found.")
    return controller.split(texts, labels)


verify_chunk_order()
