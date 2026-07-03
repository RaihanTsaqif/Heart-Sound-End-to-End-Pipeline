import argparse
import csv
import json
import os

from pipeline import HeartSoundPipeline, Thresholds, collect_audio


def flatten_row(result):
    heart = result["stages"].get("heart_sound_gate", {})
    murmur = result["stages"].get("murmur_classifier", {})
    seg = result["stages"].get("segmentation", {})
    timing = result["stages"].get("timing", {})
    return {
        "file": result["file"],
        "duration_s": round(result.get("duration_s", 0), 3),
        "final_decision": result.get("final_decision", ""),
        "heart_detected": heart.get("detected", ""),
        "p_heart": round(heart.get("p_heart", 0), 6) if heart else "",
        "murmur_detected": murmur.get("detected", ""),
        "p_murmur_present": round(murmur.get("p_present", 0), 6) if murmur else "",
        "timing": timing.get("timing", ""),
        "systolic_pct": round(timing.get("systolic_pct", 0), 3) if timing else "",
        "diastolic_pct": round(timing.get("diastolic_pct", 0), 3) if timing else "",
        "n_cycles": seg.get("n_cycles", ""),
        "runtime_s": round(result.get("runtime_s", 0), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="audio files or folders")
    parser.add_argument("--heart-threshold", type=float, default=0.5)
    parser.add_argument("--murmur-threshold", type=float, default=0.5)
    parser.add_argument("--json", help="write full JSON results")
    parser.add_argument("--csv", help="write compact CSV results")
    args = parser.parse_args()

    pipeline = HeartSoundPipeline(Thresholds(heart=args.heart_threshold, murmur=args.murmur_threshold))
    paths = collect_audio(args.paths)
    results = []
    for path in paths:
        if not os.path.exists(path):
            print(f"{path}: not found")
            continue
        result = pipeline.run(path)
        results.append(result)
        row = flatten_row(result)
        print(
            f"{row['file']}: {row['final_decision']} | "
            f"p_heart={row['p_heart']} | p_murmur={row['p_murmur_present']} | "
            f"timing={row['timing']} | runtime={row['runtime_s']}s"
        )

    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json}")
    if args.csv and results:
        parent = os.path.dirname(os.path.abspath(args.csv))
        if parent:
            os.makedirs(parent, exist_ok=True)
        rows = [flatten_row(item) for item in results]
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
