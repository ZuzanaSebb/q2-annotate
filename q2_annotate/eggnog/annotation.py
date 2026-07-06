# ----------------------------------------------------------------------------
# Copyright (c) 2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Union
from uuid import UUID

from collections import defaultdict


import pandas as pd
import skbio

from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.per_sample_sequences import MultiMAGSequencesDirFmt
from q2_types.feature_map import MAGtoContigsDirFmt
from q2_types.genome_data import (
    OrthologAnnotationDirFmt,
    Orthologs,
    SeedOrthologDirFmt,
    OrthologFileFmt,
)
from q2_types.reference_db import EggnogRefDirFmt
from q2_types.sample_data import SampleData


def _annotate_seed_orthologs_runner(
    seed_ortholog, eggnog_db, sample_label, output_loc, db_in_memory, num_cpus
):
    # at this point instead of being able to specify the type of target
    # orthologs, we want to annotate _all_.

    cmds = [
        "emapper.py",
        "-m",
        "no_search",
        "--annotate_hits_table",
        str(seed_ortholog),
        "--data_dir",
        str(eggnog_db),
        "-o",
        str(sample_label),
        "--output_dir",
        str(output_loc),
        "--cpu",
        str(num_cpus),
    ]
    if db_in_memory:
        cmds.append("--dbmem")

    subprocess.run(cmds, check=True, cwd=str(output_loc))


def _eggnog_annotate(
    eggnog_hits: SeedOrthologDirFmt,
    db: EggnogRefDirFmt,
    db_in_memory: bool = False,
    num_cpus: int = 1,
) -> OrthologAnnotationDirFmt:

    eggnog_db_fp = db.path

    result = OrthologAnnotationDirFmt()

    # run analysis
    for relpath, obj_path in eggnog_hits.seed_orthologs.iter_views(OrthologFileFmt):
        sample_label = str(relpath).rsplit(r".", 2)[0]

        _annotate_seed_orthologs_runner(
            seed_ortholog=obj_path,
            eggnog_db=eggnog_db_fp,
            sample_label=sample_label,
            output_loc=result,
            db_in_memory=db_in_memory,
            num_cpus=num_cpus,
        )

    return result


def map_eggnog(
    ctx, eggnog_hits, db, db_in_memory=False, num_cpus=1, num_partitions=None
):
    _eggnog_annotate = ctx.get_action("annotate", "_eggnog_annotate")
    collate_annotations = ctx.get_action("types", "collate_ortholog_annotations")

    if eggnog_hits.type <= SampleData[Orthologs]:
        partition_method = ctx.get_action("types", "partition_orthologs")
    else:
        raise NotImplementedError()

    (partitioned_orthologs,) = partition_method(eggnog_hits, num_partitions)

    annotations = []
    for orthologs in partitioned_orthologs.values():
        (annotation,) = _eggnog_annotate(orthologs, db, db_in_memory, num_cpus)
        annotations.append(annotation)

    (collated_annotations,) = collate_annotations(annotations)
    return collated_annotations


# this dictionary contains all the supported annotation types
# each value represents a tuple of:
# 1. original annotation column name (as it appears in the annotation table)
# 2. lambda function which will process values of that column and expand them
#   into a new series of values
extraction_methods = {
    "cog": ("COG_category", lambda x: pd.Series(list(x))),
    "kegg_ko": ("KEGG_ko", lambda x: pd.Series([i[3:] for i in x.split(",")])),
    "kegg_pathway": (
        "KEGG_Pathway",
        lambda x: pd.Series([i for i in x.split(",") if i.startswith("map")]),
    ),
    "kegg_module": ("KEGG_Module", lambda x: pd.Series(x.split(","))),
    "kegg_reaction": ("KEGG_Reaction", lambda x: pd.Series(x.split(","))),
    "brite": ("BRITE", lambda x: pd.Series(x.split(","))),
    "caz": ("CAZy", lambda x: pd.Series(x.split(","))),
    "ec": ("EC", lambda x: pd.Series(x.split(","))),
}


def _filter(data: pd.DataFrame, max_evalue: float, min_score: float) -> pd.DataFrame:
    data = data[(data["evalue"] <= max_evalue) & (data["score"] >= min_score)]
    if len(data) == 0:
        raise ValueError(
            "E-value/score filtering resulted in an empty table - "
            "please adjust your thresholds and try again."
        )
    return data


def _extract_generic(
    data: pd.DataFrame, column: str, func: callable
) -> tuple[dict, pd.Series, pd.DataFrame]:
    """
    Converts annotation data to a feature map and counts of annotation values.

    Processes the annotation DataFrame by extracting and expanding annotation
    values from the specified column, grouping them by contig ID, and counting
    their occurrences. The function applies a transformation function to expand
    annotation values and removes empty or dash values.

    Args:
        data (pd.DataFrame): The input annotation DataFrame.
        column (str): The annotation column in the DataFrame to extract.
        func (callable): The transformation function to apply to expand
            annotation values into lists.

    Returns:
        tuple[dict, pd.Series, pd.DataFrame]: A tuple containing:
            - dict: Feature map with annotation values as keys and lists of
              unique contig IDs as values.
            - pd.Series: Value counts of all annotation values.
            - pd.DataFrame: Counts of each annotation value per contig.
    """
    data = data.copy()
    data["contig_id"] = data.index.astype(str).str.rsplit("_", n=1).str[0]

    def _expand(x):
        s = func(x)
        if isinstance(s, pd.Series):
            return s.tolist()
        if isinstance(s, (list, tuple, set)):
            raise NotImplementedError(f"Unexpected return type: {type(s)}")
        return [s]

    tmp = data[["contig_id", column]].dropna(subset=[column]).copy()
    tmp["annotation_value"] = tmp[column].map(_expand)
    tmp = tmp.explode("annotation_value")

    # drop empties/dashes and any remaining missing values
    tmp = tmp[
        tmp["annotation_value"].notna() & ~tmp["annotation_value"].isin(("", "-"))
    ]

    # count annotations per contig
    contig_annotation_counts = (
        tmp.groupby("contig_id")["annotation_value"]
        .value_counts()
        .unstack(fill_value=0)
    )

    # count all the annotation values
    annotation_counts = tmp["annotation_value"].value_counts()

    # for the feature map, ensure each (annotation_value, contig_id) pair appears once
    deduplicated = tmp.drop_duplicates(subset=["annotation_value", "contig_id"])

    feature_map = (
        deduplicated.groupby("annotation_value")["contig_id"].agg(list).to_dict()
    )

    return feature_map, annotation_counts, contig_annotation_counts


def _merge_maps(maps: list[dict]) -> dict:
    """Merges a list of feature maps into a single dictionary."""
    merged = defaultdict(set)
    for d in maps:
        for k, v in d.items():
            merged[k].update(v)
    return {k: list(v) for k, v in merged.items()}


def extract_annotations(
    ortholog_annotations: OrthologAnnotationDirFmt,
    annotation: str,
    max_evalue: float = 1.0,
    min_score: float = 0.0,
) -> (pd.DataFrame, dict, pd.DataFrame):
    extract_method = extraction_methods.get(annotation)
    if not extract_method:
        raise NotImplementedError(f"Annotation '{annotation}' not supported.")
    else:
        col, func = extract_method

    annotations, feature_maps, contig_counts = [], [], []
    for _id, fp in ortholog_annotations.annotation_dict().items():
        annot_df = pd.read_csv(
            fp, sep="\t", skiprows=4, index_col=0
        )  # skip the first 4 rows as they contain comments
        # strip trailing comment rows (footer) only if present
        if annot_df.index[-3:].astype(str).str.startswith("##").all():
            annot_df = annot_df.iloc[:-3, :]
        annot_df = _filter(annot_df, max_evalue, min_score)

        # get feature map, annotation counts, and contig annotation counts
        feature_map, annot_df, contig_annot_counts = _extract_generic(
            annot_df, col, func
        )
        annot_df.name = _id

        feature_maps.append(feature_map)
        annotations.append(annot_df)
        contig_counts.append(contig_annot_counts)

    result = pd.concat(annotations, axis=1).fillna(0).T
    result.index.name = "id"
    merged_maps = _merge_maps(feature_maps)
    contig_result = pd.concat(contig_counts, axis=0).fillna(0)
    return result, dict(merged_maps), contig_result


def _validate_mag_ids(mag_ids: set, annotation_dict: dict) -> set:
    matched_ids = mag_ids & set(annotation_dict.keys())
    if not matched_ids:
        raise ValueError(
            "No annotation files matched the destination MAG IDs. "
            "Make sure the source annotations were derived from the same "
            "set of sequences as the destination."
        )
    missing = mag_ids - set(annotation_dict.keys())
    if missing:
        warnings.warn(
            f"{len(missing)} MAG(s) in the destination had no matching "
            f"annotation file in the source and will be skipped: "
            f"{', '.join(sorted(missing))}",
            UserWarning,
        )
    return matched_ids


def _copy_annotation_files(annotation_dict: dict) -> OrthologAnnotationDirFmt:
    result = OrthologAnnotationDirFmt()
    for src_path in annotation_dict.values():
        shutil.copy2(src_path, str(result.path / Path(src_path).name))
    return result


def _annotations_from_contigs(ortholog_annotations: OrthologAnnotationDirFmt) -> bool:
    """Check if annotations came from contigs (sample names) or MAGs (UUIDs)."""
    for value in ortholog_annotations.annotation_dict().keys():
        try:
            is_uuid4 = str(UUID(value, version=4)) == value
        except ValueError:
            is_uuid4 = False
        if not is_uuid4:
            return True
    return False


def _mag_ids(
    destination: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
) -> set:
    """Return the set of MAG IDs present in `destination`."""
    if isinstance(destination, MultiMAGSequencesDirFmt):
        return {
            mag_id for mags in destination.sample_dict().values() for mag_id in mags
        }
    return set(destination.feature_dict().keys())


def _build_contig_map(
    destination: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
) -> dict:
    """Build a contig map {mag_id: [contig_ids]} from destination MAG
    FASTA headers."""
    if isinstance(destination, MultiMAGSequencesDirFmt):
        mags = {
            mag_id: mag_fp
            for sample_mags in destination.sample_dict().values()
            for mag_id, mag_fp in sample_mags.items()
        }
    else:
        mags = destination.feature_dict()

    contig_map = {}
    for mag_id, mag_fp in mags.items():
        seqs = skbio.read(str(mag_fp), format="fasta", verify=False)
        contig_map[mag_id] = [x.metadata["id"] for x in seqs]
    return contig_map


def _annotate_mags_from_contigs(
    ortholog_annotations: OrthologAnnotationDirFmt,
    destination: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
    contig_map: MAGtoContigsDirFmt = None,
) -> OrthologAnnotationDirFmt:
    """Aggregate contig-level eggNOG annotations -> MAG-level annotations."""

    if contig_map is not None:
        flat_contig_map = contig_map.file.view(dict)
        valid_mag_ids = _mag_ids(destination)
        contig_map_dict = {
            k: v for k, v in flat_contig_map.items() if k in valid_mag_ids
        }
    else:
        contig_map_dict = _build_contig_map(destination)

    # reverse map: contig_id -> mag_uuid
    contig_to_mag = {
        contig_id: mag_uuid
        for mag_uuid, contig_ids in contig_map_dict.items()
        for contig_id in contig_ids
    }
    n_contigs_by_mag = {
        mag_uuid: len(contig_ids) for mag_uuid, contig_ids in contig_map_dict.items()
    }

    # read all annotation files
    frames = []
    for _id, fp in ortholog_annotations.annotation_dict().items():
        df = pd.read_csv(fp, sep="\t", skiprows=4)
        # drop trailing comment only if present
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.startswith("##")]
        frames.append(df)

    all_annotations = pd.concat(frames, ignore_index=True)
    data_cols = list(all_annotations.columns)
    col_header = "\t".join(data_cols) + "\n"

    # strip ORF suffix and map to MAG UUID
    query_col = data_cols[0]
    all_annotations["mag_uuid"] = (
        all_annotations[query_col]
        .str.replace(r"_\d+$", "", regex=True)
        .map(contig_to_mag)
    )

    matched = all_annotations.dropna(subset=["mag_uuid"])

    if matched.empty:
        raise ValueError("No annotation rows could be matched to any MAG.")

    unmatched = len(all_annotations) - len(matched)
    if unmatched > 0:
        total = len(all_annotations)
        pct = unmatched / total * 100
        warnings.warn(
            f"{unmatched} of {total} annotation row(s) ({pct:.1f}%) were on "
            "contigs not present in the contig map (e.g. unbinned contigs) "
            "and were skipped.",
            UserWarning,
        )

    result = OrthologAnnotationDirFmt()
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
            group[data_cols].to_csv(fh, sep="\t", index=False, header=False)

    print(
        f"Aggregated {len(matched)} of {len(all_annotations)} annotation "
        f"row(s) into {matched['mag_uuid'].nunique()} MAG(s); "
        f"{unmatched} row(s) skipped."
    )

    return result


def transfer_eggnog_annotations(
    ortholog_annotations: OrthologAnnotationDirFmt,
    destination: Union[MAGSequencesDirFmt, MultiMAGSequencesDirFmt],
    contig_map: MAGtoContigsDirFmt = None,
) -> OrthologAnnotationDirFmt:
    """Transfer or aggregate eggNOG annotations based on source and destination."""
    if _annotations_from_contigs(ortholog_annotations):
        # Contigs → MAGs or Contigs → Derep MAGs
        return _annotate_mags_from_contigs(
            ortholog_annotations, destination, contig_map
        )
    else:
        # MAGs → MAGs or MAGs → Derep MAGs: copy/filter by UUID
        mag_ids = _mag_ids(destination)
        annotation_dict = ortholog_annotations.annotation_dict()
        matched_ids = _validate_mag_ids(mag_ids, annotation_dict)
        return _copy_annotation_files({k: annotation_dict[k] for k in matched_ids})
