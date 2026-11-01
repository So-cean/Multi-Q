#!/usr/bin/env python3
"""Index SRPBS T1w data and create a subject-disjoint Multi-Q split.

The paired cohort comes only from SRPBS Travelling Subjects. Five independent
travelling subjects are assigned to supervised training and every remaining
subject with both selected sites is assigned to testing. The frozen population
reference comes only from healthy SRPBS OPEN participants acquired at the
selected target site. Repeated scans never increase the subject count; one
canonical T1w image per subject and site is selected deterministically.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUBJECT_KEYS = ("participant_id", "subject_id", "subject", "sub_id", "bids_id", "id")
SESSION_KEYS = ("session_id", "session", "ses_id")
SITE_KEYS = (
    "site",
    "site_id",
    "site_name",
    "center",
    "centre",
    "institution",
    "scanner_site",
)
GROUP_KEYS = (
    "group",
    "diag",
    "diagnosis",
    "diagnostic_group",
    "diagnosis_group",
    "participant_group",
    "control_patient",
)
HEALTHY_VALUES = {
    "0",
    "hc",
    "healthy",
    "healthycontrol",
    "healthycontrols",
    "control",
    "controls",
    "normalcontrol",
}
SRPBS_SITE_CODES = {
    "ATT",
    "ATV",
    "COI",
    "HUH",
    "HKH",
    "KPM",
    "KPU",
    "SWA",
    "KUT",
    "KUS",
    "UTO",
    "YC1",
    "YC2",
}


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    subject: str
    session: str
    site: str
    group: str
    path: str


def normalize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value or "").upper()


def normalize_site(value: str) -> str:
    normalized = normalize_label(value)
    return re.sub(r"^(SITE|SCANNER|CENTER|CENTRE)", "", normalized)


def normalize_subject(value: str) -> str:
    value = (value or "").strip()
    return value[4:] if value.lower().startswith("sub-") else value


def normalize_session(value: str) -> str:
    value = (value or "").strip()
    return value[4:] if value.lower().startswith("ses-") else value


def first_value(row: Mapping[str, str], keys: Sequence[str]) -> str:
    lowered = {str(key).strip().lower(): str(value).strip() for key, value in row.items() if value is not None}
    for key in keys:
        if lowered.get(key):
            return lowered[key]
    return ""


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream, delimiter="\t")]


def metadata_maps(root: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]]]:
    by_subject: Dict[str, Dict[str, str]] = {}
    by_session: Dict[Tuple[str, str], Dict[str, str]] = {}
    candidates = sorted(
        path
        for path in root.rglob("*.tsv")
        if (
            "participants" in path.name.lower()
            or "sessions" in path.name.lower()
            or re.fullmatch(r"sup\d+\.tsv", path.name.lower()) is not None
        )
    )
    for path in candidates:
        for row in read_tsv(path):
            subject = normalize_subject(first_value(row, SUBJECT_KEYS))
            session = normalize_session(first_value(row, SESSION_KEYS))
            if not subject:
                match = re.search(r"sub-([^/]+)", path.as_posix())
                subject = match.group(1) if match else ""
            if not subject:
                continue
            if session:
                by_session.setdefault((subject, session), {}).update(row)
            else:
                by_subject.setdefault(subject, {}).update(row)
    return by_subject, by_session


def entities(path: Path) -> Tuple[str, str]:
    subject = ""
    session = ""
    for part in path.parent.parts:
        if part.startswith("sub-"):
            subject = normalize_subject(part)
        elif part.startswith("ses-"):
            session = normalize_session(part)
    name = path.name
    if not subject:
        match = re.search(r"(?:^|_)sub-([^_]+)", name)
        subject = normalize_subject(match.group(1)) if match else ""
    if not session:
        match = re.search(r"(?:^|_)ses-([^_]+)", name)
        session = normalize_session(match.group(1)) if match else ""
    return subject, session


def site_from_session(session: str) -> str:
    candidate = normalize_site(session)
    return candidate if candidate and re.search(r"[A-Za-z0-9]", candidate) else ""


def site_from_subject(subject: str) -> str:
    normalized = normalize_label(subject)
    for code in sorted(SRPBS_SITE_CODES, key=len, reverse=True):
        if normalized.startswith(code):
            return code
    return ""


def is_t1w_image(path: Path) -> bool:
    lower = path.name.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        return False
    if "t1w" in lower:
        return True
    return any(parent.name.lower() == "t1" for parent in path.parents)


def index_dataset(root: Path, dataset: str) -> List[ImageRecord]:
    subject_meta, session_meta = metadata_maps(root)
    images = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and is_t1w_image(path)
    )
    records: List[ImageRecord] = []
    for path in images:
        subject, session = entities(path)
        if not subject:
            raise ValueError(f"cannot determine BIDS subject for {path}")
        merged: Dict[str, str] = {}
        merged.update(subject_meta.get(subject, {}))
        merged.update(session_meta.get((subject, session), {}))
        site = first_value(merged, SITE_KEYS) or site_from_session(session) or site_from_subject(subject)
        group = first_value(merged, GROUP_KEYS)
        records.append(
            ImageRecord(
                dataset=dataset,
                subject=subject,
                session=session,
                site=normalize_site(site),
                group=normalize_label(group),
                path=str(path),
            )
        )
    return records


def is_healthy(record: ImageRecord) -> bool:
    return normalize_label(record.group).lower() in HEALTHY_VALUES


def canonical_by_subject_site(records: Iterable[ImageRecord]) -> Dict[Tuple[str, str], ImageRecord]:
    grouped: Dict[Tuple[str, str], List[ImageRecord]] = {}
    for record in records:
        grouped.setdefault((record.subject, record.site), []).append(record)
    return {key: sorted(values, key=lambda item: (item.session, item.path))[0] for key, values in grouped.items()}


def site_summary(records: Sequence[ImageRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in sorted({record.dataset for record in records}):
        subset = [record for record in records if record.dataset == dataset]
        for site in sorted({record.site for record in subset}):
            at_site = [record for record in subset if record.site == site]
            rows.append(
                {
                    "dataset": dataset,
                    "site": site,
                    "subjects": len({record.subject for record in at_site}),
                    "images": len(at_site),
                    "healthy_subjects": len({record.subject for record in at_site if is_healthy(record)}),
                    "unknown_group_subjects": len({record.subject for record in at_site if not record.group}),
                }
            )
    return rows


def candidate_pairs(ts_records: Sequence[ImageRecord], open_records: Sequence[ImageRecord]) -> List[Dict[str, object]]:
    ts = canonical_by_subject_site(ts_records)
    ts_sites = sorted({record.site for record in ts_records if record.site})
    open_sites = {record.site for record in open_records if record.site}
    rows: List[Dict[str, object]] = []
    for target in sorted(open_sites.intersection(ts_sites)):
        healthy_reference = {
            record.subject for record in open_records if record.site == target and is_healthy(record)
        }
        for source in ts_sites:
            if source == target:
                continue
            paired = sorted(
                subject for subject in {record.subject for record in ts_records}
                if (subject, source) in ts and (subject, target) in ts
            )
            rows.append(
                {
                    "source_site": source,
                    "target_site": target,
                    "paired_subjects": len(paired),
                    "healthy_reference_subjects": len(healthy_reference),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["paired_subjects"]),
            -int(row["healthy_reference_subjects"]),
            str(row["target_site"]),
            str(row["source_site"]),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_site(requested: Optional[str], available: Iterable[str], label: str) -> Optional[str]:
    if requested is None:
        return None
    normalized = normalize_site(requested)
    if normalized not in set(available):
        raise ValueError(f"unknown {label} {requested!r}; available={sorted(set(available))}")
    return normalized


def build_split(
    ts_records: Sequence[ImageRecord],
    open_records: Sequence[ImageRecord],
    output: Path,
    train_subjects: int,
    seed: int,
    source_site: Optional[str],
    target_site: Optional[str],
    train_subject_ids: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    candidates = candidate_pairs(ts_records, open_records)
    write_csv(
        output / "pair_candidates.csv",
        candidates,
        ("source_site", "target_site", "paired_subjects", "healthy_reference_subjects"),
    )
    sites = {record.site for record in ts_records if record.site}
    source_site = resolve_site(source_site, sites, "source site")
    target_site = resolve_site(target_site, sites, "target site")
    if source_site is None or target_site is None:
        eligible = [
            row for row in candidates
            if int(row["paired_subjects"]) > train_subjects
            and int(row["healthy_reference_subjects"]) > 0
        ]
        if not eligible:
            raise ValueError(
                "no automatic site pair has both held-out travelling subjects and healthy OPEN references; "
                "inspect pair_candidates.csv and provide --source-site/--target-site"
            )
        source_site = str(eligible[0]["source_site"])
        target_site = str(eligible[0]["target_site"])
    if source_site == target_site:
        raise ValueError("source and target sites must differ")

    canonical = canonical_by_subject_site(ts_records)
    paired_subjects = sorted(
        subject for subject in {record.subject for record in ts_records}
        if (subject, source_site) in canonical and (subject, target_site) in canonical
    )
    if len(paired_subjects) <= train_subjects:
        raise ValueError(
            f"site pair {source_site}->{target_site} has {len(paired_subjects)} subjects; "
            f"need more than {train_subjects} for an independent test set"
        )
    if train_subject_ids is None:
        rng = random.Random(seed)
        shuffled = list(paired_subjects)
        rng.shuffle(shuffled)
        train_ids = sorted(shuffled[:train_subjects])
    else:
        requested = [normalize_subject(value) for value in train_subject_ids]
        if len(requested) != train_subjects or len(set(requested)) != train_subjects:
            raise ValueError(
                f"Expected {train_subjects} unique explicit training subjects, got {requested}"
            )
        unknown = sorted(set(requested).difference(paired_subjects))
        if unknown:
            raise ValueError(f"Explicit training subjects are not paired: {unknown}")
        train_ids = sorted(requested)
    test_ids = sorted(set(paired_subjects).difference(train_ids))

    def pair_rows(subjects: Sequence[str]) -> List[Dict[str, str]]:
        return [
            {
                "subject": subject,
                "source_site": source_site,
                "source_path": canonical[(subject, source_site)].path,
                "target_site": target_site,
                "target_path": canonical[(subject, target_site)].path,
            }
            for subject in subjects
        ]

    train_rows = pair_rows(train_ids)
    test_rows = pair_rows(test_ids)
    reference_records = sorted(
        (
            record for record in canonical_by_subject_site(open_records).values()
            if record.site == target_site and is_healthy(record)
        ),
        key=lambda record: (record.subject, record.path),
    )
    if not reference_records:
        raise ValueError(f"no healthy OPEN reference images found for target site {target_site}")
    reference_ids = {record.subject for record in reference_records}
    if set(train_ids).intersection(test_ids):
        raise AssertionError("travelling-subject leakage between train and test")
    if {row["source_path"] for row in train_rows + test_rows}.intersection(
        record.path for record in reference_records
    ):
        raise AssertionError("paired image leaked into the population reference")

    pair_fields = ("subject", "source_site", "source_path", "target_site", "target_path")
    write_csv(output / "train_pairs.csv", train_rows, pair_fields)
    write_csv(output / "test_pairs.csv", test_rows, pair_fields)
    write_csv(
        output / "population_reference.csv",
        [asdict(record) for record in reference_records],
        tuple(ImageRecord.__dataclass_fields__),
    )
    result: Dict[str, object] = {
        "status": "ready",
        "seed": seed,
        "split_assignment": "explicit" if train_subject_ids is not None else "seeded_random",
        "source_site": source_site,
        "target_site": target_site,
        "train_subjects": train_ids,
        "test_subjects": test_ids,
        "train_subject_count": len(train_ids),
        "test_subject_count": len(test_ids),
        "population_reference_subject_count": len(reference_ids),
        "population_reference_image_count": len(reference_records),
        "independence_checks": {
            "train_test_subjects_disjoint": True,
            "paired_reference_paths_disjoint": True,
            "reference_dataset": "SRPBS_OPEN",
            "paired_dataset": "SRPBS_TS",
            "repeated_scans_count_as_subject_repeats": True,
        },
    }
    (output / "split.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--travelling-root", type=Path, required=True)
    parser.add_argument("--open-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-subjects", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-site")
    parser.add_argument("--target-site")
    parser.add_argument(
        "--train-subject-ids",
        help="Comma-separated explicit training subjects; must match --train-subjects.",
    )
    args = parser.parse_args()

    ts_records = index_dataset(args.travelling_root.resolve(), "SRPBS_TS")
    open_records = index_dataset(args.open_root.resolve(), "SRPBS_OPEN")
    if not ts_records or not open_records:
        raise ValueError(
            f"missing T1w images: travelling={len(ts_records)}, open={len(open_records)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    all_records = sorted(ts_records + open_records, key=lambda record: (record.dataset, record.path))
    write_csv(
        args.output / "t1w_index.csv",
        [asdict(record) for record in all_records],
        tuple(ImageRecord.__dataclass_fields__),
    )
    summaries = site_summary(all_records)
    write_csv(
        args.output / "site_summary.csv",
        summaries,
        ("dataset", "site", "subjects", "images", "healthy_subjects", "unknown_group_subjects"),
    )
    result = build_split(
        ts_records,
        open_records,
        args.output,
        args.train_subjects,
        args.seed,
        args.source_site,
        args.target_site,
        [value.strip() for value in args.train_subject_ids.split(",") if value.strip()]
        if args.train_subject_ids
        else None,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
