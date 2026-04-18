"""
OLMo OCR 2 Pipeline (LM Studio API)
====================================
Converts a PDF into page images + markdown using OLMo OCR 2 (7B)
served via LM Studio's OpenAI-compatible API.

Outputs:
    - data/pdf_images/page_XXX.png   (rendered page images)
    - data/markdown/page_XXX.md      (per-page markdown)
    - corpus.json                    (all pages)
    - sample.json                    (first N pages)
"""

import base64
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import re

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ── olmocr utilities ─────────────────────────────────────────────────────────
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# ── project config ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    API_BASE_URL,
    API_MODEL,
    CORPUS_JSON,
    MARKDOWN_DIR,
    MAX_NEW_TOKENS,
    MAX_RETRIES,
    NUM_WORKERS,
    PDF_IMAGES_DIR,
    PDF_PATH,
    RETRY_DELAY,
    SAMPLE_JSON,
    SAMPLE_PAGES,
    TARGET_LONGEST_IMAGE_DIM,
    TEMPERATURE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 0 — Pre-flight checks
# ═══════════════════════════════════════════════════════════════════════════════

def preflight_checks() -> None:
    if not PDF_PATH.exists():
        sys.exit(f"ERROR: PDF not found at {PDF_PATH}")

    # Verify LM Studio API is reachable
    client = OpenAI(base_url=API_BASE_URL, api_key="lm-studio")
    try:
        client.models.list()
        print(f"LM Studio API: {API_BASE_URL} — connected")
        print(f"Model: {API_MODEL}")
        print(f"Workers: {NUM_WORKERS}")
    except Exception as e:
        sys.exit(f"ERROR: Cannot reach LM Studio API at {API_BASE_URL}\n  {e}")

    PDF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage A — PDF → page images
# ═══════════════════════════════════════════════════════════════════════════════

def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF using pypdfium2 (bundled with olmocr)."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def render_pages(pdf_path: Path, num_pages: int) -> list[Path]:
    """Render each PDF page to a PNG in pdf_images/."""
    image_paths: list[Path] = []
    print(f"\n{'='*60}")
    print(f"Stage A — Rendering {num_pages} PDF pages to images")
    print(f"{'='*60}")

    poppler_ready = bool(shutil.which("pdfinfo") and shutil.which("pdftoppm"))
    pdf_doc = None

    if not poppler_ready:
        import pypdfium2 as pdfium

        print("  Poppler tools not found (pdfinfo/pdftoppm). Falling back to pypdfium2 renderer.")
        pdf_doc = pdfium.PdfDocument(str(pdf_path))

    for page_num in tqdm(range(1, num_pages + 1), desc="Rendering pages"):
        if poppler_ready:
            try:
                img_b64 = render_pdf_to_base64png(
                    str(pdf_path), page_num, target_longest_image_dim=TARGET_LONGEST_IMAGE_DIM
                )
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_bytes))
            except FileNotFoundError:
                import pypdfium2 as pdfium

                print("  Poppler tools became unavailable during rendering; switching to pypdfium2 fallback.")
                poppler_ready = False
                if pdf_doc is None:
                    pdf_doc = pdfium.PdfDocument(str(pdf_path))
                page = pdf_doc[page_num - 1]
                width, height = page.get_size()
                scale = TARGET_LONGEST_IMAGE_DIM / max(width, height)
                bitmap = page.render(scale=scale)
                img = bitmap.to_pil()
        else:
            page = pdf_doc[page_num - 1]
            width, height = page.get_size()
            scale = TARGET_LONGEST_IMAGE_DIM / max(width, height)
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()

        out_path = PDF_IMAGES_DIR / f"page_{page_num:03d}.png"
        img.save(out_path)
        image_paths.append(out_path)

    if pdf_doc is not None:
        pdf_doc.close()

    print(f"  Saved {len(image_paths)} images to {PDF_IMAGES_DIR}")
    return image_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Stage B — Create API client
# ═══════════════════════════════════════════════════════════════════════════════

def create_client() -> OpenAI:
    """Create an OpenAI-compatible client pointing at LM Studio."""
    print(f"\n{'='*60}")
    print(f"Stage B — Connecting to LM Studio API")
    print(f"{'='*60}")
    print(f"  Endpoint: {API_BASE_URL}")
    print(f"  Model:    {API_MODEL}")
    print(f"  Workers:  {NUM_WORKERS}")

    client = OpenAI(base_url=API_BASE_URL, api_key="lm-studio")
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# Stage C — OCR inference (per page)
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_page(
    client: OpenAI,
    pdf_path: Path,
    page_num: int,
    image_path: Path,
) -> str:
    """Run OCR on a single page via the LM Studio API and return the markdown text."""
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # Build prompt with anchor text from the PDF text layer
    try:
        anchor_text = get_anchor_text(str(pdf_path), page_num, pdf_engine="pdfreport", target_length=4000)
    except Exception:
        anchor_text = ""
    prompt = build_finetuning_prompt(anchor_text) + (
        "\n\nOutput requirements:\n"
        "- Return valid Markdown only.\n"
        "- Preserve math as LaTeX with $...$ for inline and $$...$$ for block equations.\n"
        "- Do not escape LaTeX backslashes unless required by Markdown syntax.\n"
        "- Do not wrap the entire answer in markdown code fences.\n"
    )

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    if not response.choices:
        print(f"  [WARN] Empty response for page {page_num}: {response}")
        return ""
    return response.choices[0].message.content or ""


def ocr_page_with_retry(
    client: OpenAI,
    pdf_path: Path,
    page_num: int,
    image_path: Path,
) -> str:
    """Call ocr_page with retries and backoff on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return ocr_page(client, pdf_path, page_num, image_path)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  [ERROR] Page {page_num} failed after {MAX_RETRIES} attempts: {e}")
                return ""
            wait = RETRY_DELAY * attempt
            print(f"  [RETRY] Page {page_num} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            print(f"           Waiting {wait}s for LM Studio to recover...")
            time.sleep(wait)
    return ""


def normalize_markdown_latex(text: str) -> str:
    """Normalize model output to plain markdown while preserving LaTeX math delimiters."""
    cleaned = text.strip()

    # Some model responses are JSON wrappers with a natural_text payload.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            natural_text = parsed.get("natural_text")
            if isinstance(natural_text, str) and natural_text.strip():
                cleaned = natural_text.strip()
    except json.JSONDecodeError:
        # Fallback for JSON-like payloads with invalid escapes (for example \$).
        match = re.search(r'"natural_text"\s*:\s*"(.*)"\s*}\s*$', cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(1)
            candidate = candidate.replace(r"\$", "$")
            candidate = candidate.replace(r"\n", "\n")
            candidate = candidate.replace(r'\"', '"')
            if candidate.strip():
                cleaned = candidate.strip()

    # Remove accidental top-level markdown fences added by the model.
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()

    # Normalize escaped dollar signs used as math delimiters.
    cleaned = cleaned.replace(r"\$", "$")

    # Turn \(...\) and \[...\] into markdown-compatible math delimiters.
    cleaned = re.sub(r"\\\((.+?)\\\)", r"$\1$", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\\\[(.+?)\\\]", r"$$\n\1\n$$", cleaned, flags=re.DOTALL)

    return cleaned.strip() + "\n"


def run_ocr(client: OpenAI, pdf_path: Path, image_paths: list[Path]) -> list[dict]:
    """Run OCR on all pages, optionally in parallel (controlled by NUM_WORKERS).
    Skips pages that already have a markdown file (resume support)."""
    print(f"\n{'='*60}")
    print(f"Stage C — Running OCR on {len(image_paths)} pages ({NUM_WORKERS} workers)")
    print(f"{'='*60}")

    results: dict[int, str] = {}
    source_name = pdf_path.name

    # Resume: load already-processed pages
    skipped = 0
    for i, img_path in enumerate(image_paths):
        idx = i + 1
        md_path = MARKDOWN_DIR / f"page_{idx:03d}.md"
        if md_path.exists() and md_path.stat().st_size > 0:
            results[idx] = md_path.read_text(encoding="utf-8")
            skipped += 1

    if skipped:
        print(f"  Resuming: {skipped} pages already done, {len(image_paths) - skipped} remaining")

    # Filter to only pages that need processing
    todo = [(i + 1, img_path) for i, img_path in enumerate(image_paths) if (i + 1) not in results]

    def _process_page(idx: int, img_path: Path) -> tuple[int, str]:
        md_text = ocr_page_with_retry(client, pdf_path, idx, img_path)
        return idx, md_text

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for idx, img_path in todo:
            future = executor.submit(_process_page, idx, img_path)
            futures[future] = idx

        with tqdm(total=len(todo), desc="OCR pages", initial=0) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    md_text = future.result()[1]
                except Exception as e:
                    print(f"  [ERROR] Page {idx} failed: {e}")
                    md_text = ""
                md_text = normalize_markdown_latex(md_text)

                md_path = MARKDOWN_DIR / f"page_{idx:03d}.md"
                md_path.write_text(md_text, encoding="utf-8")

                results[idx] = md_text
                pbar.update(1)
                tqdm.write(f"  page {idx:03d}: {len(md_text)} chars")

    # Build pages list in order
    pages = []
    for idx in sorted(results.keys()):
        md_text = results[idx]
        pages.append(
            {
                "source": source_name,
                "page": idx,
                "char_count": len(md_text),
                "text": md_text,
            }
        )

    print(f"  Saved {len(pages)} markdown files to {MARKDOWN_DIR}")
    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# Stage D — Save JSON outputs
# ═══════════════════════════════════════════════════════════════════════════════

def save_json_outputs(pages: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("Stage D — Saving JSON outputs")
    print(f"{'='*60}")

    CORPUS_JSON.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  corpus.json: {len(pages)} pages -> {CORPUS_JSON}")

    sample = pages[:SAMPLE_PAGES]
    SAMPLE_JSON.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  sample.json: {len(sample)} pages -> {SAMPLE_JSON}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    preflight_checks()

    num_pages = get_page_count(PDF_PATH)
    print(f"  PDF: {PDF_PATH.name} — {num_pages} pages")

    # Stage A — render pages
    image_paths = render_pages(PDF_PATH, num_pages)

    # Stage B — connect to API
    client = create_client()

    # Stage C — OCR
    pages = run_ocr(client, PDF_PATH, image_paths)

    # Stage D — save outputs
    save_json_outputs(pages)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
