"""Local-evidence helpers for open-set source discovery reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


class SourceDiscoveryError(ValueError):
    """Raised when recovery evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SplitEvidence:
    """Identifiers used to detect overlap with the frozen V2 splits."""

    source_ids: frozenset[str]
    perceptual_group_ids: frozenset[str]
    sha256_digests: frozenset[str]


def build_perceptual_groups(
    image_names: Iterable[str], near_duplicate_pairs: Iterable[Mapping[str, str]]
) -> dict[str, str]:
    """Build the deterministic Dataset B near-duplicate groups used by V2."""
    names = set(image_names)
    if not names:
        return {}
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            raise SourceDiscoveryError(
                f"Near-duplicate pair references an unknown image: {left}, {right}"
            )
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in near_duplicate_pairs:
        try:
            union(pair["left_image"], pair["right_image"])
        except KeyError as error:
            raise SourceDiscoveryError(
                "Near-duplicate pair is missing left_image or right_image"
            ) from error
    roots = sorted({find(name) for name in names})
    root_ids = {root: f"group_{index:04d}" for index, root in enumerate(roots, 1)}
    return {name: root_ids[find(name)] for name in names}


def make_split_evidence(
    split_rows: Iterable[Mapping[str, str]], image_hashes: Mapping[str, str]
) -> SplitEvidence:
    """Collect frozen-split source, group, and exact-content identifiers."""
    sources: set[str] = set()
    groups: set[str] = set()
    digests: set[str] = set()
    for row in split_rows:
        try:
            source_id = row["source_image_id"]
            group_id = row["perceptual_group_id"]
        except KeyError as error:
            raise SourceDiscoveryError(
                "Frozen split row lacks source_image_id or perceptual_group_id"
            ) from error
        if source_id in sources:
            raise SourceDiscoveryError(
                f"Duplicate source ID across splits: {source_id}"
            )
        sources.add(source_id)
        groups.add(group_id)
        digest = image_hashes.get(source_id)
        if digest is None:
            raise SourceDiscoveryError(
                f"Missing image hash for split source: {source_id}"
            )
        digests.add(digest)
    return SplitEvidence(frozenset(sources), frozenset(groups), frozenset(digests))


def frozen_split_overlap(
    *,
    source_id: str,
    perceptual_group_id: str,
    sha256: str,
    evidence: SplitEvidence,
) -> tuple[bool, str]:
    """Return whether a recovery candidate overlaps V2 and the matching reason."""
    reasons: list[str] = []
    if source_id in evidence.source_ids:
        reasons.append("source_image_id")
    if perceptual_group_id in evidence.perceptual_group_ids:
        reasons.append("perceptual_group_id")
    if sha256 in evidence.sha256_digests:
        reasons.append("exact_sha256")
    return bool(reasons), ";".join(reasons)


def eligible_counts_by_class(
    recovery_rows: Iterable[Mapping[str, str]],
) -> dict[str, dict[str, int]]:
    """Count eligible photographs and conservative dependency groups per label."""
    images: dict[str, int] = defaultdict(int)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in recovery_rows:
        if row.get("eligible_for_unseen_review") != "yes":
            continue
        label = row["newly_proposed_unseen_label"]
        images[label] += 1
        groups[label].add(row["conservative_dependency_group"])
    return {
        label: {
            "eligible_images": images[label],
            "independent_groups": len(groups[label]),
        }
        for label in sorted(images)
    }
