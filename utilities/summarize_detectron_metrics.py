#!/usr/bin/env python3
"""Advanced Detectron2 / detection metrics summarizer.

See docstring for details – scans *metrics.json files, aggregates AP metrics,
classifies runs, and prints a ranked table + CSV.
"""

import argparse
import csv
import json
import os
import fnmatch
import re
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize detection metrics logs (Detectron2-style) and rank runs."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to search recursively.",
    )
    parser.add_argument(
        "--metrics-pattern",
        type=str,
        default="*metrics.json",
        help="Filename pattern to match (supports wildcards, e.g. '*metrics.json'). "
             "Default: *metrics.json",
    )
    parser.add_argument(
        "--ap-key",
        type=str,
        default="bbox/AP",
        help='Primary metric key to treat as main AP (default: "bbox/AP").',
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Only keep runs whose FULL PATH contains ALL of these substrings (case-insensitive).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Drop runs whose FULL PATH contains ANY of these substrings (case-insensitive).",
    )
    parser.add_argument(
        "--ap-threshold",
        type=float,
        default=0.4,
        help="AP threshold above which a run is classified as KEEP/EVAL_KEEP (default: 0.4).",
    )
    parser.add_argument(
        "--min-iters",
        type=int,
        default=500,
        help="Minimum last iteration to consider a *training* run complete (default: 500).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top runs to print. Use 0 or negative to show all.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path and run identity helpers
# ---------------------------------------------------------------------------

def path_matches_filters(
    path: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> bool:
    """Check include/exclude filters on the full path (case-insensitive)."""
    path_lower = path.lower()

    if include:
        for s in include:
            if s.lower() not in path_lower:
                return False

    if exclude:
        for s in exclude:
            if s.lower() in path_lower:
                return False

    return True


def find_metrics_files(
    root: str,
    metrics_pattern: str,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
) -> List[str]:
    """Find all files whose filename matches the given pattern (e.g. *metrics.json)."""
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fnmatch.fnmatch(fname, metrics_pattern):
                full_path = os.path.join(dirpath, fname)
                if path_matches_filters(full_path, include, exclude):
                    matches.append(full_path)
    return sorted(matches)


def infer_run_identity(metrics_path: str) -> Tuple[str, str, str]:
    """Infer (run_dir, run_name, file_type) from metrics path.

    - training log: .../run_xxx/metrics.json
        -> run_dir = that folder
        -> run_name = folder name
        -> file_type = "train_log"

    - eval file: .../coco_eval/lr0002_b1024_metrics.json
        -> run_dir = parent of coco_eval
        -> run_name = "lr0002_b1024" (filename without _metrics.json)
        -> file_type = "eval_file"

    - anything else:
        -> run_dir = dirname
        -> run_name = dirname basename
        -> file_type = "other"
    """
    fname = os.path.basename(metrics_path)
    parent_dir = os.path.basename(os.path.dirname(metrics_path))

    if fname == "metrics.json":
        run_dir = os.path.dirname(metrics_path)
        run_name = parent_dir
        file_type = "train_log"
    elif fname.endswith("_metrics.json"):
        run_name = fname[: -len("_metrics.json")]
        if parent_dir.lower() in ("coco_eval", "eval", "evaluation"):
            run_dir = os.path.dirname(os.path.dirname(metrics_path))
        else:
            run_dir = os.path.dirname(metrics_path)
        file_type = "eval_file"
    else:
        run_dir = os.path.dirname(metrics_path)
        run_name = parent_dir
        file_type = "other"

    return run_dir, run_name, file_type


# ---------------------------------------------------------------------------
# JSON loading and metric extraction
# ---------------------------------------------------------------------------

def load_json_records(path: str) -> List[Dict[str, Any]]:
    """Load a metrics file as a list of JSON records."""
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return []

    # Try whole-file JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback: JSON lines
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def extract_bbox_metrics(record: Dict[str, Any], ap_key: str):
    """Try to extract bbox metrics (AP, AP50, AP75, APs, APm, APl) from one record."""
    # flattened Detectron-style keys
    ap = record.get(ap_key)
    ap50 = record.get("bbox/AP50")
    ap75 = record.get("bbox/AP75")
    aps = record.get("bbox/APs")
    apm = record.get("bbox/APm")
    apl = record.get("bbox/APl")

    # nested under "bbox"
    bbox = record.get("bbox")
    if isinstance(bbox, dict):
        ap = bbox.get("AP") if ap is None else ap
        ap50 = bbox.get("AP50") if ap50 is None else ap50
        ap75 = bbox.get("AP75") if ap75 is None else ap75
        aps = bbox.get("APs") if aps is None else aps
        apm = bbox.get("APm") if apm is None else apm
        apl = bbox.get("APl") if apl is None else apl

    # flat COCO keys
    if ap is None and "AP" in record:
        ap = record["AP"]
    if ap50 is None and "AP50" in record:
        ap50 = record["AP50"]
    if ap75 is None and "AP75" in record:
        ap75 = record["AP75"]
    if aps is None and "APs" in record:
        aps = record["APs"]
    if apm is None and "APm" in record:
        apm = record["APm"]
    if apl is None and "APl" in record:
        apl = record["APl"]

    # alt AP keys (torch etc.)
    if ap is None:
        for k in ["mAP", "map", "val_map", "val_mAP", "val_ap"]:
            if k in record and isinstance(record[k], (int, float)):
                ap = record[k]
                break

    # alt AP50 keys
    if ap50 is None:
        for k in ["map_50", "mAP_50", "AP@0.50", "AP_50", "val_map_50"]:
            if k in record and isinstance(record[k], (int, float)):
                ap50 = record[k]
                break

    return ap, ap50, ap75, aps, apm, apl


def parse_run_name_tags(run_name: str) -> Dict[str, Any]:
    """Heuristically parse hyperparam tags from run_name."""
    tags: Dict[str, Any] = {}
    name = run_name.lower()

    m = re.search(r"(r50|r101|resnet50|resnet101|res50|res101)", name)
    if m:
        tags["backbone"] = m.group(1)

    m = re.search(r"(?:^|[_\-])b(\d+)(?:$|[_\-])", name)
    if m:
        tags["batch_tag"] = m.group(1)

    m = re.search(r"lr([0-9eE\.\-]+)", name)
    if m:
        tags["lr_tag"] = m.group(1)

    m = re.search(r"nms([0-9\.]+)", name)
    if m:
        tags["nms_tag"] = m.group(1)

    m = re.search(r"(?:^|[_\-])s([0-9\.]+)", name)
    if m:
        tags["score_tag"] = m.group(1)

    if "clutter" in name:
        tags["experiment"] = "clutter_mix"
    elif "boost" in name:
        tags["experiment"] = "boost_ap"
    elif "nobg" in name or "no_bg" in name:
        tags["experiment"] = "no_bg"
    elif "default" in name:
        tags["experiment"] = "default"
    else:
        tags["experiment"] = "other"

    return tags


# ---------------------------------------------------------------------------
# Core summarization logic
# ---------------------------------------------------------------------------

def summarize_single_metrics_file(
    metrics_path: str,
    root: str,
    ap_key: str = "bbox/AP",
) -> Dict[str, Any]:
    run_dir, run_name, file_type = infer_run_identity(metrics_path)
    run_rel_dir = os.path.relpath(run_dir, root)

    records = load_json_records(metrics_path)

    best_ap = best_ap50 = best_ap75 = best_aps = best_apm = best_apl = None
    best_iter = None

    first_ap = first_iter = None

    last_ap = last_ap50 = last_ap75 = last_aps = last_apm = last_apl = None
    last_iter: Optional[int] = None

    num_eval_events = 0

    for rec in records:
        it = rec.get("iteration")
        if isinstance(it, int):
            last_iter = it

        ap_val, ap50_val, ap75_val, aps_val, apm_val, apl_val = extract_bbox_metrics(rec, ap_key)
        if ap_val is None:
            continue

        num_eval_events += 1

        if first_ap is None:
            first_ap = ap_val
            first_iter = it if isinstance(it, int) else None

        last_ap = ap_val
        last_ap50 = ap50_val
        last_ap75 = ap75_val
        last_aps = aps_val
        last_apm = apm_val
        last_apl = apl_val

        if (best_ap is None) or (ap_val > best_ap):
            best_ap = ap_val
            best_ap50 = ap50_val
            best_ap75 = ap75_val
            best_aps = aps_val
            best_apm = apm_val
            best_apl = apl_val
            best_iter = it if isinstance(it, int) else best_iter

    ap_gain_first_last = None
    if isinstance(first_ap, (int, float)) and isinstance(last_ap, (int, float)):
        ap_gain_first_last = last_ap - first_ap

    ap_drop_after_best = None
    if isinstance(best_ap, (int, float)) and isinstance(last_ap, (int, float)):
        ap_drop_after_best = best_ap - last_ap

    best_iter_fraction = None
    if isinstance(best_iter, int) and isinstance(last_iter, int) and last_iter > 0:
        best_iter_fraction = best_iter / float(last_iter)

    summary: Dict[str, Any] = {
        "metrics_path": metrics_path,
        "metrics_rel_path": os.path.relpath(metrics_path, root),
        "run_dir": run_dir,
        "run_rel_dir": run_rel_dir,
        "run_name": run_name,
        "file_type": file_type,
        "best_ap": best_ap,
        "best_ap50": best_ap50,
        "best_ap75": best_ap75,
        "best_aps": best_aps,
        "best_apm": best_apm,
        "best_apl": best_apl,
        "best_iter": best_iter,
        "best_iter_fraction": best_iter_fraction,
        "first_ap": first_ap,
        "first_iter": first_iter,
        "last_ap": last_ap,
        "last_ap50": last_ap50,
        "last_ap75": last_ap75,
        "last_aps": last_aps,
        "last_apm": last_apm,
        "last_apl": last_apl,
        "last_iter": last_iter,
        "ap_gain_first_last": ap_gain_first_last,
        "ap_drop_after_best": ap_drop_after_best,
        "num_eval_events": num_eval_events,
        "training_phase": None,
    }

    summary.update(parse_run_name_tags(run_name))
    return summary


def classify_run(
    summary: Dict[str, Any],
    ap_threshold: float,
    min_iters: int,
) -> str:
    best_ap = summary.get("best_ap")
    last_iter = summary.get("last_iter")
    num_eval_events = summary.get("num_eval_events", 0)
    file_type = summary.get("file_type", "train_log")

    if best_ap is None or num_eval_events == 0:
        return "NO_EVAL"

    if file_type == "eval_file":
        return "EVAL_KEEP" if best_ap >= ap_threshold else "EVAL_REDO"

    if last_iter is None or last_iter < min_iters:
        return "INCOMPLETE"

    if best_ap >= ap_threshold:
        return "KEEP"
    return "REDO_CANDIDATE"


def label_training_phase(summary: Dict[str, Any], min_iters: int) -> str:
    """Coarse label of training evolution for train logs."""
    if summary.get("file_type") != "train_log":
        return "N/A"

    num_eval = summary.get("num_eval_events", 0)
    last_iter = summary.get("last_iter")
    best_iter_fraction = summary.get("best_iter_fraction")
    ap_gain_first_last = summary.get("ap_gain_first_last")
    ap_drop_after_best = summary.get("ap_drop_after_best")

    if last_iter is None or num_eval == 0:
        return "UNKNOWN"

    if last_iter < min_iters:
        return "EARLY_STOP"

    if num_eval < 2 or best_iter_fraction is None:
        return "UNKNOWN"

    # Peak near the end
    if best_iter_fraction >= 0.9:
        if isinstance(ap_gain_first_last, (int, float)) and ap_gain_first_last > 0.5:
            return "STILL_IMPROVING"
        if isinstance(ap_drop_after_best, (int, float)) and ap_drop_after_best < 0.3:
            return "PEAK_AT_END"
        return "PEAK_LATE"
    # peak late-ish
    if best_iter_fraction >= 0.6:
        return "PEAK_LATE"
    # peak mid
    if best_iter_fraction >= 0.3:
        return "PEAK_MID"
    # peak early
    return "PEAK_EARLY"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(summaries: List[Dict[str, Any]], csv_path: str) -> None:
    if not summaries:
        return

    fieldnames = [
        "run_name",
        "run_dir",
        "run_rel_dir",
        "metrics_path",
        "metrics_rel_path",
        "file_type",
        "status",
        "training_phase",
        "best_ap",
        "best_ap50",
        "best_ap75",
        "best_aps",
        "best_apm",
        "best_apl",
        "best_iter",
        "best_iter_fraction",
        "first_ap",
        "first_iter",
        "last_ap",
        "last_ap50",
        "last_ap75",
        "last_aps",
        "last_apm",
        "last_apl",
        "last_iter",
        "ap_gain_first_last",
        "ap_drop_after_best",
        "num_eval_events",
        "backbone",
        "batch_tag",
        "lr_tag",
        "nms_tag",
        "score_tag",
        "experiment",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k) for k in fieldnames}
            writer.writerow(row)


def print_summary_table(
    sorted_summaries: List[Dict[str, Any]],
    top_k: int,
) -> None:
    if not sorted_summaries:
        print("No runs found after filtering.")
        return

    if top_k <= 0 or top_k > len(sorted_summaries):
        top_k = len(sorted_summaries)

    print("\n=== Top runs by best AP ===")
    print(f"(Showing top {top_k} of {len(sorted_summaries)} total)\n")

    header = (
        f"{'Rank':>4} | {'AP':>6} | {'AP50':>6} | {'Evals':>5} | "
        f"{'BestIt':>6} | {'LastIt':>6} | {'Gain':>7} | {'Drop':>7} | "
        f"{'Phase':>14} | {'Type':>10} | {'Status':>14} | "
        f"{'BB':>5} | {'B':>4} | {'LR':>10} | {'Exp':>9} | RelPath"
    )
    print(header)
    print("-" * len(header))

    for idx, s in enumerate(sorted_summaries[:top_k], start=1):
        ap = s.get("best_ap")
        ap50 = s.get("best_ap50")
        best_iter = s.get("best_iter")
        last_iter = s.get("last_iter")
        phase = s.get("training_phase", "N/A")
        status = s.get("status")
        file_type = s.get("file_type")
        backbone = s.get("backbone", "")
        batch_tag = s.get("batch_tag", "")
        lr_tag = s.get("lr_tag", "")
        exp = s.get("experiment", "")

        evals = s.get("num_eval_events")
        gain = s.get("ap_gain_first_last")
        drop = s.get("ap_drop_after_best")

        ap_str = f"{ap:.3f}" if isinstance(ap, (int, float)) else "  N/A"
        ap50_str = f"{ap50:.3f}" if isinstance(ap50, (int, float)) else "  N/A"
        best_it_str = f"{best_iter}" if isinstance(best_iter, int) else "  N/A"
        last_it_str = f"{last_iter}" if isinstance(last_iter, int) else "  N/A"
        evals_str = f"{evals}" if isinstance(evals, int) else "  N/A"
        gain_str = f"{gain:+.3f}" if isinstance(gain, (int, float)) else "   N/A"
        drop_str = f"{drop:+.3f}" if isinstance(drop, (int, float)) else "   N/A"

        print(
            f"{idx:>4} | {ap_str:>6} | {ap50_str:>6} | {evals_str:>5} | "
            f"{best_it_str:>6} | {last_it_str:>6} | {gain_str:>7} | {drop_str:>7} | "
            f"{phase:>14} | {file_type:>10} | {status:>14} | "
            f"{str(backbone):>5} | {str(batch_tag):>4} | {str(lr_tag):>10} | {str(exp):>9} | "
            f"{s.get('run_rel_dir')}"
        )

    status_counts: Dict[str, int] = {}
    for s in sorted_summaries:
        st = s.get("status", "UNKNOWN")
        status_counts[st] = status_counts.get(st, 0) + 1

    print("\n=== Status counts ===")
    for st, cnt in sorted(status_counts.items(), key=lambda x: x[0]):
        print(f"{st:>14}: {cnt}")


def print_group_best(
    sorted_summaries: List[Dict[str, Any]],
    rank_map: Dict[str, int],
    key: str,
    label: str,
) -> None:
    groups: Dict[str, Tuple[float, str, str]] = {}
    for s in sorted_summaries:
        val = s.get(key)
        ap = s.get("best_ap")
        if not val or not isinstance(ap, (int, float)):
            continue
        if val not in groups or ap > groups[val][0]:
            groups[val] = (ap, s.get("metrics_rel_path", ""), s.get("run_name", ""))

    if not groups:
        return

    print(f"\n=== Best AP per {label} ===")
    for val, (ap, mrel, run_name) in sorted(groups.items(), key=lambda x: x[1][0], reverse=True):
        rank = rank_map.get(mrel, "?")
        print(f"{label}={val:>10} | AP={ap:.3f} | rank={rank} | run={run_name}")


def print_recommended_runs(
    sorted_summaries: List[Dict[str, Any]],
    rank_map: Dict[str, int],
) -> None:
    train_keep = [
        s for s in sorted_summaries
        if s.get("file_type") == "train_log"
        and s.get("status") == "KEEP"
        and isinstance(s.get("best_ap"), (int, float))
    ]
    if train_keep:
        best_train = max(train_keep, key=lambda s: s["best_ap"])
        rank = rank_map.get(best_train.get("metrics_rel_path", ""), "?")
        print("\n>>> Suggested canonical TRAIN baseline:")
        print(
            f"    rank={rank} | {best_train.get('run_name')} "
            f"(AP={best_train['best_ap']:.3f}, backbone={best_train.get('backbone')}, "
            f"batch={best_train.get('batch_tag')}, lr={best_train.get('lr_tag')}, "
            f"experiment={best_train.get('experiment')}, path={best_train.get('run_rel_dir')})"
        )

    eval_keep = [
        s for s in sorted_summaries
        if s.get("file_type") == "eval_file"
        and s.get("status") == "EVAL_KEEP"
        and isinstance(s.get("best_ap"), (int, float))
    ]
    if eval_keep:
        best_eval = max(eval_keep, key=lambda s: s["best_ap"])
        rank = rank_map.get(best_eval.get("metrics_rel_path", ""), "?")
        print("\n>>> Suggested canonical EVAL baseline (threshold / checkpoint combo):")
        print(
            f"    rank={rank} | {best_eval.get('run_name')} "
            f"(AP={best_eval['best_ap']:.3f}, backbone={best_eval.get('backbone')}, "
            f"batch={best_eval.get('batch_tag')}, lr={best_eval.get('lr_tag')}, "
            f"experiment={best_eval.get('experiment')}, path={best_eval.get('run_rel_dir')})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    metrics_files = find_metrics_files(
        root=args.root,
        metrics_pattern=args.metrics_pattern,
        include=args.include,
        exclude=args.exclude,
    )

    if not metrics_files:
        print("No metrics files found with the given filters.")
        return

    print(f"Found {len(metrics_files)} metrics files. Parsing...")

    summaries: List[Dict[str, Any]] = []
    for path in metrics_files:
        s = summarize_single_metrics_file(path, root=args.root, ap_key=args.ap_key)
        s["status"] = classify_run(s, ap_threshold=args.ap_threshold, min_iters=args.min_iters)
        s["training_phase"] = label_training_phase(s, min_iters=args.min_iters)
        summaries.append(s)

    # sort once by best AP
    sorted_summaries = sorted(
        summaries,
        key=lambda s: float(s["best_ap"]) if isinstance(s.get("best_ap"), (int, float)) else float("-inf"),
        reverse=True,
    )

    # map metrics_rel_path -> rank index
    rank_map: Dict[str, int] = {}
    for idx, s in enumerate(sorted_summaries, start=1):
        mrel = s.get("metrics_rel_path")
        if mrel is not None:
            rank_map[mrel] = idx

    csv_path = os.path.join(args.root, "metrics_summary.csv")
    save_csv(sorted_summaries, csv_path)
    print(f"\nSaved summary CSV to: {csv_path}")

    print_summary_table(sorted_summaries, top_k=args.top_k)

    print_group_best(sorted_summaries, rank_map, key="backbone", label="backbone")
    print_group_best(sorted_summaries, rank_map, key="batch_tag", label="batch")
    print_group_best(sorted_summaries, rank_map, key="lr_tag", label="lr")
    print_group_best(sorted_summaries, rank_map, key="experiment", label="experiment")

    print_recommended_runs(sorted_summaries, rank_map)


if __name__ == "__main__":
    main()
#  python summarize_detectron_metrics.py   --root "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt"   --ap-threshold 0.40   --min-iters 15000   --top-k 41