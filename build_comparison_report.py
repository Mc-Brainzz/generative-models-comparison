from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class MetricRow:
    method: str
    ssim: str = ""
    psnr: str = ""
    train_time: str = ""
    infer_time: str = ""
    notes: str = ""
    source: str = ""


def _clean_cell(value: str) -> str:
    value = value.strip()
    value = value.replace("**", "")
    value = value.replace("`", "")
    value = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", value)
    return value.strip()


def _extract_number(text: str) -> str:
    if not text:
        return ""
    raw = text.replace("~", "").replace("dB", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    return match.group(0) if match else ""


def _is_separator_row(line: str) -> bool:
    compact = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return compact == ""


def parse_markdown_tables(md_text: str, source_name: str) -> list[MetricRow]:
    lines = md_text.splitlines()
    rows: list[MetricRow] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            i += 1
            continue

        header_cells = [_clean_cell(c) for c in line.strip("|").split("|")]
        lower_header = [h.lower() for h in header_cells]

        if "method" not in lower_header:
            i += 1
            continue

        method_idx = lower_header.index("method")

        def find_col(*keywords: str) -> int | None:
            for idx, col in enumerate(lower_header):
                if any(k in col for k in keywords):
                    return idx
            return None

        ssim_idx = find_col("ssim")
        psnr_idx = find_col("psnr")
        train_idx = find_col("train")
        infer_idx = find_col("infer")
        notes_idx = find_col("note")

        if ssim_idx is None and psnr_idx is None:
            i += 1
            continue

        j = i + 1
        if j < len(lines) and lines[j].strip().startswith("|") and _is_separator_row(lines[j]):
            j += 1

        while j < len(lines):
            row_line = lines[j].strip()
            if not row_line.startswith("|"):
                break
            if _is_separator_row(row_line):
                j += 1
                continue

            cells = [_clean_cell(c) for c in row_line.strip("|").split("|")]
            if method_idx >= len(cells):
                j += 1
                continue

            method = cells[method_idx]
            ssim = _extract_number(cells[ssim_idx]) if ssim_idx is not None and ssim_idx < len(cells) else ""
            psnr = _extract_number(cells[psnr_idx]) if psnr_idx is not None and psnr_idx < len(cells) else ""
            train_time = cells[train_idx] if train_idx is not None and train_idx < len(cells) else ""
            infer_time = cells[infer_idx] if infer_idx is not None and infer_idx < len(cells) else ""
            notes = cells[notes_idx] if notes_idx is not None and notes_idx < len(cells) else ""

            if method:
                rows.append(
                    MetricRow(
                        method=method,
                        ssim=ssim,
                        psnr=psnr,
                        train_time=train_time,
                        infer_time=infer_time,
                        notes=notes,
                        source=source_name,
                    )
                )
            j += 1

        i = j

    return rows


def canonical_method_name(name: str) -> str:
    n = name.lower()
    if "bicubic" in n:
        return "Bicubic"
    if "sdedit" in n:
        return "SDEdit"
    if "ddib" in n:
        return "DDIB"
    if "cyclegan" in n:
        return "CycleGAN SR"
    if "deterministic" in n and ("flow" in n or "ep" in n):
        return "Flow Matching Deterministic (Unpaired)"
    if "stochastic" in n and ("flow" in n or "ep" in n):
        return "Flow Matching Stochastic (Unpaired)"
    if "flow matching" in n:
        return "Flow Matching"
    if "vae" in n:
        return "VAE"
    if "gan" in n:
        return "GAN"
    if "paired training" in n:
        return "Flow Matching Paired"
    if "truly unpaired" in n:
        return "Flow Matching Unpaired (OT)"
    return name


def choose_best_rows(rows: Iterable[MetricRow]) -> dict[str, MetricRow]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        canon = canonical_method_name(row.method)
        grouped.setdefault(canon, []).append(row)

    selected: dict[str, MetricRow] = {}
    for canon, entries in grouped.items():
        def score(e: MetricRow) -> tuple[float, float]:
            ssim = float(e.ssim) if e.ssim else -1.0
            psnr = float(e.psnr) if e.psnr else -1.0
            return (ssim, psnr)

        best = sorted(entries, key=score, reverse=True)[0]
        best.method = canon
        selected[canon] = best

    return selected


def build_report(root: Path, selected: dict[str, MetricRow], sources: list[Path], output: Path) -> None:
    report_order = [
        "Bicubic",
        "SDEdit",
        "DDIB",
        "CycleGAN SR",
        "Flow Matching Stochastic (Unpaired)",
        "Flow Matching Deterministic (Unpaired)",
    ]

    image_files = sorted((root / "Super_resolution").glob("results*.png"))

    lines: list[str] = []
    lines.append("# Auto Comparison Report (Generated)")
    lines.append("")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Sources Parsed")
    for src in sources:
        rel = src.relative_to(root).as_posix()
        lines.append(f"- {rel}")
    lines.append("")
    lines.append("## Quantitative Comparison")
    lines.append("")
    lines.append("| Method | SSIM | PSNR | Train Time | Infer Time | Notes | Source |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    for method in report_order:
        row = selected.get(method)
        if row is None:
            lines.append(f"| {method} |  |  |  |  | pending run |  |")
            continue
        lines.append(
            "| {m} | {ssim} | {psnr} | {train} | {infer} | {notes} | {source} |".format(
                m=method,
                ssim=row.ssim,
                psnr=row.psnr,
                train=row.train_time,
                infer=row.infer_time,
                notes=row.notes or "",
                source=row.source,
            )
        )

    lines.append("")
    lines.append("## Key Extracted Findings")
    det = selected.get("Flow Matching Deterministic (Unpaired)")
    stoch = selected.get("Flow Matching Stochastic (Unpaired)")
    bic = selected.get("Bicubic")

    if det and stoch and det.ssim and stoch.ssim:
        gain = float(det.ssim) - float(stoch.ssim)
        lines.append(f"- Deterministic vs stochastic SSIM delta: {gain:+.4f}")
    if det and bic and det.ssim and bic.ssim:
        gap = float(det.ssim) - float(bic.ssim)
        lines.append(f"- Deterministic vs bicubic SSIM gap: {gap:+.4f}")
    lines.append("- Use this as a controlled benchmark claim, then validate on an external dataset.")

    lines.append("")
    lines.append("## Available Visuals")
    if image_files:
        for img in image_files:
            lines.append(f"- {img.relative_to(root).as_posix()}")
    else:
        lines.append("- No result images found.")

    output.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: dict[str, MetricRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ssim", "psnr", "train_time", "infer_time", "notes", "source"])
        for method in sorted(rows.keys()):
            row = rows[method]
            writer.writerow([method, row.ssim, row.psnr, row.train_time, row.infer_time, row.notes, row.source])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison report from existing repository findings.")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument("--output", type=str, default="COMPARISON_REPORT_AUTO.md", help="Output markdown report")
    parser.add_argument("--csv", type=str, default="comparison_metrics_auto.csv", help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source_files = [
        root / "Super_resolution" / "FINAL_README.md",
        root / "Super_resolution" / "TRULY_UNPAIRED_README.md",
        root / "Super_resolution" / "FLOW_MATCHING_README.md",
    ]

    all_rows: list[MetricRow] = []
    used_sources: list[Path] = []

    for src in source_files:
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        rows = parse_markdown_tables(text, src.relative_to(root).as_posix())
        if rows:
            all_rows.extend(rows)
            used_sources.append(src)

    if not all_rows:
        raise SystemExit("No metrics tables found in source markdown files.")

    selected = choose_best_rows(all_rows)

    report_path = (root / args.output).resolve()
    csv_path = (root / args.csv).resolve()

    build_report(root, selected, used_sources, report_path)
    write_csv(selected, csv_path)

    print(f"Generated report: {report_path}")
    print(f"Generated csv:    {csv_path}")


if __name__ == "__main__":
    main()
