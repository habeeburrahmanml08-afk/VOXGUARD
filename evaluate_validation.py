"""Evaluate labeled WAV files using VOXGUARD's production preprocessing."""

import argparse
import hashlib
import json
from pathlib import Path


SUPPORTED_SUFFIXES = {".wav", ".wave"}


def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def evaluate(dataset_dir):
    dataset_dir = Path(dataset_dir)
    bona_fide_files = sorted(
        path for path in (dataset_dir / "bonafide").rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ) if (dataset_dir / "bonafide").exists() else []
    spoof_files = sorted(
        path for path in (dataset_dir / "spoof").rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ) if (dataset_dir / "spoof").exists() else []
    files = [(path, 1) for path in bona_fide_files] + [(path, 0) for path in spoof_files]
    if not bona_fide_files or not spoof_files:
        raise ValueError(
            "Add labeled WAV files to both folders: "
            "<dataset>\\bonafide and <dataset>\\spoof"
        )

    # Load only after checking arguments/folders, so --help stays quick.
    from backend_api import _prepare_uploaded_audio
    from predict import predict_audio_array

    groups = {}
    for path, truth in files:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        groups.setdefault(digest, []).append((path, truth, payload))

    results = []
    unique_files = []
    conflicting_duplicate_groups = 0
    duplicate_files_excluded = 0
    for entries in groups.values():
        labels = {truth for _, truth, _ in entries}
        first_path = entries[0][0]
        if len(entries) > 1:
            duplicate_files_excluded += len(entries) - 1
        if len(labels) > 1:
            conflicting_duplicate_groups += 1
            for path, truth, _ in entries:
                results.append({
                    "file": str(path.relative_to(dataset_dir)),
                    "ground_truth": "bonafide" if truth == 1 else "spoof",
                    "prediction": None,
                    "status": "exact duplicate appears under conflicting labels; excluded",
                })
            continue
        unique_files.append(entries[0])
        for path, truth, _ in entries[1:]:
            results.append({
                "file": str(path.relative_to(dataset_dir)),
                "ground_truth": "bonafide" if truth == 1 else "spoof",
                "prediction": None,
                "status": f"exact duplicate of {first_path.relative_to(dataset_dir)}; excluded",
            })

    accepted = 0
    correct = 0
    bonafide_total = 0
    spoof_total = 0
    bonafide_scored = 0
    spoof_scored = 0
    bonafide_false_alarms = 0
    spoof_misses = 0
    inference_times = []
    matrix = {
        "bonafide": {"predicted_bonafide": 0, "predicted_spoof": 0},
        "spoof": {"predicted_bonafide": 0, "predicted_spoof": 0},
    }

    for path, truth, payload in unique_files:
        ground_truth = "bonafide" if truth == 1 else "spoof"
        if truth == 1:
            bonafide_total += 1
        else:
            spoof_total += 1
        entry = {"file": str(path.relative_to(dataset_dir)), "ground_truth": ground_truth}
        try:
            window, quality, usable, duration = _prepare_uploaded_audio(payload)
            entry["audio_duration_seconds"] = round(duration, 3)
            entry["audio_quality"] = quality
            if not usable:
                entry.update({"prediction": None, "status": quality.get("reason", "insufficient audio")})
                results.append(entry)
                continue
            output = predict_audio_array(window)
            predicted = int(output["predicted_index"])
            prediction = "bonafide" if predicted == 1 else "spoof"
            inference_ms = float(output["inference_ms"])
            inference_times.append(inference_ms)
            accepted += 1
            correct += int(predicted == truth)
            if truth == 1:
                bonafide_scored += 1
            else:
                spoof_scored += 1
            matrix[ground_truth][f"predicted_{prediction}"] += 1
            if truth == 1 and predicted == 0:
                bonafide_false_alarms += 1
            if truth == 0 and predicted == 1:
                spoof_misses += 1
            entry.update({
                "prediction": prediction,
                "correct": predicted == truth,
                "inference_ms": round(inference_ms, 2),
            })
        except Exception as exc:
            entry.update({"prediction": None, "status": f"could not process: {type(exc).__name__}: {exc}"})
        results.append(entry)

    total = len(unique_files)
    report = {
        "note": "Offline validation only; metrics describe these files and are not a guarantee for future calls.",
        "dataset_dir": str(dataset_dir.resolve()),
        "window_samples": 64600,
        "sample_rate": 16000,
        "input_files": len(files),
        "unique_files_evaluated": total,
        "duplicate_files_excluded": duplicate_files_excluded,
        "conflicting_duplicate_groups_excluded": conflicting_duplicate_groups,
        "accepted_files": accepted,
        "unscored_files": total - accepted,
        "coverage": _rate(accepted, total),
        "accuracy_on_scored_files": _rate(correct, accepted),
        "bonafide_scored_files": bonafide_scored,
        "spoof_scored_files": spoof_scored,
        "bonafide_false_alarm_rate_on_scored_files": _rate(bonafide_false_alarms, bonafide_scored),
        "spoof_miss_rate_on_scored_files": _rate(spoof_misses, spoof_scored),
        "confusion_matrix": matrix,
        "mean_inference_ms": round(sum(inference_times) / len(inference_times), 2) if inference_times else None,
        "files": results,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="validation", help="folder containing bonafide\\ and spoof\\ WAV subfolders")
    args = parser.parse_args()
    try:
        report = evaluate(args.dataset)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
