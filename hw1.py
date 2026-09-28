#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Design: a *parallel extraction + deterministic aggregation* chain.

    Stage 1 (vision, one call per receipt): the ``deepseek-v4-flash-vision-exp``
    model transcribes each receipt into a strict JSON record holding the item
    lines, the negative discount lines, the ``小計``/``SUBTOTAL`` line, the
    ``ROUNDING`` line and the genuinely charged payment line, together with two
    self-consistency checks it must satisfy before answering.

    Stage 2 (pure Python, no LLM): every receipt is read twice, agreeing readings
    are accepted directly, and a disagreement or a failed check triggers one
    repair round in which the failed check is quoted back to the model; the
    surviving readings are settled by majority vote. The settled records are then
    summed with ``Decimal`` arithmetic, so the model never has to add anything up,
    and a receipt that still cannot be read is salvaged from whichever fields did
    parse instead of crashing the run.

    Returns a small dict bundle consumed by :func:`answer_queries`.
    """
    import os

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set. Put your key in .env as "
            "DEEPSEEK_API_KEY=sk-... (the file is gitignored) and run again."
        )

    system_prompt = (
        "You are a meticulous receipt-reading engine for Hong Kong supermarket "
        "receipts (PARKnSHOP / Fusion, Wellcome, ...) that mix Traditional "
        "Chinese and English. You transcribe the printed numbers exactly as they "
        "appear, you never guess, and you always answer with one JSON object."
    )

    extraction_prompt = """Read every line of this receipt image and return ONE JSON object, nothing else.

Required fields (plain numbers, no "$", no thousands separators, keep negative signs):
- "item_lines": array of every positive product price line above the 小計/SUBTOTAL line.
- "discount_lines": array of every negative line above the 小計/SUBTOTAL line, such as packaging-damage 包裝變形, Buy N Save, "% OFF", coupon, member/app discount. Do NOT put ROUNDING in this array.
- "subtotal": the value printed on the 小計 / SUBTOTAL line.
- "rounding": the value printed on the ROUNDING line (use 0 if there is none).
- "total_paid": the amount actually charged, i.e. the payment line directly AFTER ROUNDING, such as OCTOPUS / 八達通 / 現金 / CASH / EPS. Never use 餘額 (remaining balance) or 找續 (change) for this field.

Consistency rules you MUST satisfy before answering:
1. sum(item_lines) - sum(abs(discount_lines)) must equal subtotal (tolerance 0.05).
2. subtotal + rounding must equal total_paid (tolerance 0.005).
If a rule fails, re-read the image line by line and correct the arrays first.

Reply with the JSON object only: no markdown fences, no commentary."""

    repair_prompt = """Your previous JSON failed the consistency check below. Re-read the receipt image line by line and reply with the corrected JSON object only, no commentary.

Failures found:
{failures}

Most common causes: a negative discount line above 小計/SUBTOTAL was missed or wrongly typed as positive, ROUNDING was mixed into the discount lines, or the payment line was confused with 餘額 / 找續 / a second tender line."""

    model = ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp"),
        temperature=0,
        max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "16000")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                    {"type": "text", "text": extraction_prompt},
                ],
            ),
        ]
    )

    return {
        "model": model,
        "prompt": prompt,
        "repair_prompt": repair_prompt,
        "extract": prompt | model,
    }


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    import os

    from langchain_core.messages import AIMessage, HumanMessage

    debug = bool(os.environ.get("HW1_DEBUG"))
    model = chain["model"]
    prompt = chain["prompt"]
    extract = chain["extract"]
    repair_prompt = chain["repair_prompt"]
    cents = Decimal("0.01")
    zero = Decimal("0")

    def as_number(value: Any) -> Decimal | None:
        """Coerce whatever the model printed into a Decimal, or None."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float, Decimal)):
            text = str(value)
        elif isinstance(value, str):
            text = re.sub(r"[^0-9.\-]", "", value.replace(",", ""))
        else:
            return None
        if not re.search(r"\d", text):
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    def as_number_list(value: Any) -> list[Decimal]:
        if isinstance(value, dict):
            value = list(value.values())
        if not isinstance(value, (list, tuple)):
            return []
        return [number for number in (as_number(item) for item in value) if number is not None]

    def as_payload(response: Any) -> dict[str, Any] | None:
        """Pull the first JSON object out of a model response."""
        text = response_text(response)
        if not text:
            return None
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None
        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), None)
        return data if isinstance(data, dict) else None

    def as_fields(payload: dict[str, Any]) -> dict[str, Any]:
        """Normalise one receipt record and derive the two aggregate values."""
        item_lines = as_number_list(payload.get("item_lines"))
        discount_lines = [abs(value) for value in as_number_list(payload.get("discount_lines"))]
        fields = {
            "items_total": sum(item_lines, zero) if item_lines else None,
            "discount_total": sum(discount_lines, zero),
            "subtotal": as_number(payload.get("subtotal")),
            "rounding": as_number(payload.get("rounding")) or zero,
            "total_paid": as_number(payload.get("total_paid")),
        }
        return fields

    def failures(fields: dict[str, Any]) -> list[str]:
        """Return the list of failed self-consistency checks (empty when OK)."""
        found = []
        subtotal, rounding, paid = fields["subtotal"], fields["rounding"], fields["total_paid"]
        if subtotal is None or paid is None:
            return ["the JSON did not contain both a subtotal and a total_paid number"]
        if abs(subtotal + rounding - paid) > Decimal("0.005"):
            return_note = f"subtotal {subtotal} + rounding {rounding} = {subtotal + rounding}, but total_paid is {paid}"
            found.append(return_note)
        items = fields["items_total"]
        if items is not None and abs(items - fields["discount_total"] - subtotal) > Decimal("0.05"):
            found.append(
                f"sum(item_lines) {items} - discount_total {fields['discount_total']} = "
                f"{items - fields['discount_total']}, but subtotal is {subtotal}"
            )
        return found

    def salvage(fields: dict[str, Any]) -> dict[str, Decimal]:
        """Best-effort per-receipt contribution when checks keep failing."""
        subtotal, rounding, paid = fields["subtotal"], fields["rounding"], fields["total_paid"]
        if subtotal is None and paid is not None:
            subtotal = paid - rounding
        if paid is None and subtotal is not None:
            paid = subtotal + rounding
        if paid is None:
            return {"paid": zero, "without": zero}
        if subtotal is None:
            subtotal = paid - rounding
        return {"paid": paid, "without": subtotal + fields["discount_total"]}

    def contribution(fields: dict[str, Any]) -> tuple[Decimal, Decimal] | None:
        """The only two numbers one receipt contributes to the final answers."""
        if not fields:
            return None
        part = salvage(fields)
        return (part["paid"], part["without"])

    def re_read(url: str, reasons: list[str]) -> dict[str, Any]:
        """One repair round: hand the failed check back to the model."""
        try:
            messages = prompt.format_messages(image_url=url)
            messages.append(AIMessage(content="I could not confirm that reading of the receipt."))
            messages.append(HumanMessage(content=repair_prompt.format(failures="\n".join(f"- {r}" for r in reasons))))
            payload = as_payload(model.invoke(messages))
        except Exception:
            payload = None
        return as_fields(payload) if payload else {}

    # ---- Stage 1: two independent reads of every receipt, fanned out in parallel.
    urls = [image_data_url(path) for path in images]
    reads_per_receipt = 2
    inputs = [{"image_url": url} for url in urls for _ in range(reads_per_receipt)]
    try:
        responses = list(extract.batch(inputs, config={"max_concurrency": 4}))
    except Exception:
        responses = []
        for item in inputs:
            try:
                responses.append(extract.invoke(item))
            except Exception:
                responses.append(None)

    candidates: list[list[dict[str, Any]]] = []
    for index in range(len(urls)):
        window = responses[index * reads_per_receipt : (index + 1) * reads_per_receipt]
        readings = []
        for response in window:
            payload = as_payload(response)
            readings.append(as_fields(payload) if payload else {})
        candidates.append(readings)

    # ---- Stage 2: resolve each receipt by agreement; conflicted ones get one
    # repair round, then a majority vote that prefers readings passing both checks.
    records: list[dict[str, Any]] = []
    for index, url in enumerate(urls):
        pair = candidates[index]
        agreed = {contribution(candidate) for candidate in pair if candidate and not failures(candidate)}
        agreed.discard(None)
        if len(agreed) != 1:
            reasons = next((failures(c) for c in pair if c and failures(c)), None)
            if reasons is None:
                reasons = ["two independent readings of the same image disagreed on the amounts"]
            repaired = re_read(url, reasons)
            if repaired:
                pair = pair + [repaired]
        pool = [c for c in pair if c and not failures(c)] or [c for c in pair if c]
        tally: dict[Any, list[dict[str, Any]]] = {}
        for candidate in pool:
            tally.setdefault(contribution(candidate), []).append(candidate)
        records.append(max(tally.values(), key=len)[0] if tally else {})
        if debug:
            print(f"[hw1] {Path(images[index]).name} readings={[dict(c) for c in pair]} picked={records[-1]}")

    # ---- Stage 3: deterministic aggregation (the model never does the maths).
    paid_total = zero
    without_total = zero
    for fields in records:
        part = salvage(fields) if fields else {"paid": zero, "without": zero}
        paid_total += part["paid"]
        without_total += part["without"]

    return {
        QUERY_1: f"HK${paid_total.quantize(cents)}",
        QUERY_2: f"HK${without_total.quantize(cents)}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
