# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import shutil
import warnings
from pathlib import Path
from typing import Union

import pandas as pd
import skbio
from qiime2.core.type import Properties

from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.genome_data import GenomeData, NOG, OrthologAnnotationDirFmt
from q2_types.per_sample_sequences import MultiMAGSequencesDirFmt


def _validate_mag_ids(mag_ids: set, annotations: dict) -> set:
    matched_ids = mag_ids & set(annotations.keys())
    if not matched_ids:
        raise ValueError(
            "No annotation files matched the destination MAG IDs. "
            "Make sure the source annotations were derived from the same "
            "set of sequences as the destination."
        )
    missing = mag_ids - set(annotations.keys())
    if missing:
        warnings.warn(
            f"{len(missing)} MAG(s) in the destination had no matching "
            f"annotation file in the source and will be skipped: "
            f"{', '.join(sorted(missing))}",
            UserWarning,
        )
    return matched_ids


def _copy_annotation_files(annotations: dict) -> OrthologAnnotationDirFmt:
    result = OrthologAnnotationDirFmt()
    for src_path in annotations.values():
        shutil.copy2(src_path, str(result.path / Path(src_path).name))
    return result


def _get_mag_ids(
    destination_sequences: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
) -> set:
    """Return the set of MAG IDs present in `destination_sequences`."""
    if isinstance(destination_sequences, MultiMAGSequencesDirFmt):
        return {
            mag_id
            for mags in destination_sequences.sample_dict().values()
            for mag_id in mags
        }
    return set(destination_sequences.feature_dict().keys())


def _build_contig_map(
    destination_sequences: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
) -> dict:
    """Build a contig map {mag_id: [contig_ids]} from destination MAG
    FASTA headers."""
    if isinstance(destination_sequences, MultiMAGSequencesDirFmt):
        mags = {
            mag_id: mag_fp
            for sample_mags in destination_sequences.sample_dict().values()
            for mag_id, mag_fp in sample_mags.items()
        }
    else:
        mags = destination_sequences.feature_dict()

    contig_map = {}
    for mag_id, mag_fp in mags.items():
        seqs = skbio.read(str(mag_fp), format="fasta", verify=False)
        contig_map[mag_id] = [x.metadata["id"] for x in seqs]
    return contig_map


def _resolve_contig_map(
    destination_sequences: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
    source_contig_map: dict = None,
) -> dict:
    """Return filtered source contig map, or build one from destination."""
    if source_contig_map is not None:
        valid_mag_ids = _get_mag_ids(destination_sequences)
        return {k: v for k, v in source_contig_map.items() if k in valid_mag_ids}
    return _build_contig_map(destination_sequences)


def _reverse_contig_map(contig_map: dict) -> tuple[dict, dict]:
    """Invert MAG->contigs mapping and count contigs per MAG."""
    contig_to_mag = {
        contig_id: mag_uuid
        for mag_uuid, contig_ids in contig_map.items()
        for contig_id in contig_ids
    }
    n_contigs_by_mag = {
        mag_uuid: len(contig_ids) for mag_uuid, contig_ids in contig_map.items()
    }
    return contig_to_mag, n_contigs_by_mag


def _load_annotation_rows(
    source_annotations: OrthologAnnotationDirFmt,
) -> pd.DataFrame:
    frames = []
    for _id, fp in source_annotations.annotation_dict().items():
        df = pd.read_csv(
            fp,
            sep="\t",
            skiprows=4,
            dtype=str,
            keep_default_na=False,
        )
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.startswith("##")]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _map_rows_to_mag_ids(
    all_annotations: pd.DataFrame, contig_to_mag: dict
) -> pd.DataFrame:
    mapped = all_annotations.copy()
    query_col = mapped.columns[0]
    mapped["mag_uuid"] = (
        mapped[query_col].str.replace(r"_\d+$", "", regex=True).map(contig_to_mag)
    )
    return mapped


def _require_matched_annotation_rows(
    all_annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    matched = all_annotations.dropna(subset=["mag_uuid"])
    if matched.empty:
        raise ValueError("No annotation rows could be matched to any MAG.")
    total = len(all_annotations)
    return matched, total


def _warn_unmatched_annotation_rows(unmatched: int, total: int) -> None:
    if unmatched > 0:
        pct = unmatched / total * 100
        warnings.warn(
            f"{unmatched} of {total} annotation row(s) ({pct:.1f}%) could not be "
            "matched to any MAG in the destination sequences "
            "and were skipped.",
            UserWarning,
        )


def _write_grouped_annotations(
    matched: pd.DataFrame, n_contigs_by_mag: dict
) -> OrthologAnnotationDirFmt:
    result = OrthologAnnotationDirFmt()
    data_cols = [col for col in matched.columns if col != "mag_uuid"]
    col_header = "\t".join(data_cols) + "\n"

    for mag_uuid, group in matched.groupby("mag_uuid"):
        out_fp = result.path / f"{mag_uuid}.emapper.annotations"
        n_contigs = n_contigs_by_mag.get(mag_uuid, 0)
        n_rows = len(group)
        with open(out_fp, "w") as fh:
            fh.write("## Transferred using transfer_eggnog_annotations (q2-annotate)\n")
            fh.write("## Source: contig-level annotations\n")
            fh.write(f"## MAG: {mag_uuid} | contigs: {n_contigs} | rows: {n_rows}\n")
            fh.write("##\n")
            fh.write(col_header)
            group[data_cols].to_csv(
                fh,
                sep="\t",
                index=False,
                header=False,
            )

    return result


def _transfer_annotations_from_contigs(
    source_annotations: OrthologAnnotationDirFmt,
    destination_sequences: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
    source_contig_map: dict = None,
) -> OrthologAnnotationDirFmt:
    """Transfer contig-level eggNOG annotations -> MAG-level annotations."""
    contig_map = _resolve_contig_map(destination_sequences, source_contig_map)
    contig_to_mag, n_contigs_by_mag = _reverse_contig_map(contig_map)
    all_annotations = _load_annotation_rows(source_annotations)
    all_annotations = _map_rows_to_mag_ids(all_annotations, contig_to_mag)
    matched, total = _require_matched_annotation_rows(all_annotations)
    unmatched = total - len(matched)
    _warn_unmatched_annotation_rows(unmatched, total)

    result = _write_grouped_annotations(matched, n_contigs_by_mag)

    print(
        f"Transferred {len(matched)} of {total} annotation "
        f"row(s) into {matched['mag_uuid'].nunique()} MAG(s); "
        f"{unmatched} row(s) skipped."
    )

    return result


def _transfer_annotations_from_mags(
    source_annotations: OrthologAnnotationDirFmt,
    destination_sequences: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
) -> OrthologAnnotationDirFmt:
    """Transfer MAG-level eggNOG annotations for MAGs matching the destination."""
    mag_ids = _get_mag_ids(destination_sequences)
    annotations = source_annotations.annotation_dict()
    matched_ids = _validate_mag_ids(mag_ids, annotations)
    return _copy_annotation_files({k: annotations[k] for k in matched_ids})


def transfer_eggnog_annotations(
    ctx,
    source_annotations,
    destination_sequences,
    source_contig_map=None,
):
    """Transfer eggNOG annotations based on source and destination."""
    if source_annotations.type <= GenomeData[NOG % Properties("contigs")]:
        transfer_action = ctx.get_action(
            "annotate", "_transfer_annotations_from_contigs"
        )
        args = (source_annotations, destination_sequences, source_contig_map)
    else:
        if source_contig_map is not None:
            warnings.warn(
                "`source_contig_map` is only valid for contig-level source "
                "annotations and will be ignored.",
                UserWarning,
            )
        transfer_action = ctx.get_action("annotate", "_transfer_annotations_from_mags")
        args = (source_annotations, destination_sequences)

    (transferred_annotations,) = transfer_action(*args)
    return (transferred_annotations,)
