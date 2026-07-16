# ----------------------------------------------------------------------------
# Copyright (c) 2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import filecmp
from pathlib import Path

import pandas as pd
from qiime2 import Artifact
from qiime2.core.type import Properties
from qiime2.plugin.testing import TestPluginBase

from q2_annotate.eggnog.transfer import (
    _annotate_mags_from_contigs,
    _copy_mag_annotations,
    _load_annotation_rows,
    _map_rows_to_mag_ids,
    _require_matched_annotation_rows,
    _resolve_contig_map,
    _reverse_contig_map,
    _warn_unmatched_annotation_rows,
    _write_grouped_annotations,
)
from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.feature_map import MAGtoContigsDirFmt
from q2_types.genome_data import GenomeData, NOG, OrthologAnnotationDirFmt
from q2_types.per_sample_sequences import MultiMAGSequencesDirFmt

MAG1 = "1e9ffc02-0847-4f2c-b1e2-3965a4a78b15"
MAG2 = "62e07985-2556-435c-9e02-e7f94b8df07d"
CONTIG_MAG1 = "11111111-1111-4111-8111-111111111111"
CONTIG_MAG2 = "22222222-2222-4222-8222-222222222222"


def _build_fasta_content(contig_ids=None):
    """Builds placeholder FASTA content, one record per contig ID."""
    contig_ids = ["placeholder"] if contig_ids is None else contig_ids
    return "".join(f">{contig_id}\nACGT\n" for contig_id in contig_ids)


def _build_mag_dirfmt(root, mag_ids, contig_ids_by_mag=None):
    """Builds a `MAGSequencesDirFmt` with one flat FASTA file per MAG."""
    root.mkdir()
    for mag_id in mag_ids:
        contig_ids = None if contig_ids_by_mag is None else contig_ids_by_mag[mag_id]
        (root / f"{mag_id}.fasta").write_text(_build_fasta_content(contig_ids))
    return MAGSequencesDirFmt(root, mode="r")


def _build_multi_mag_dirfmt(root, mag_ids, contig_ids_by_mag=None):
    """Builds a `MultiMAGSequencesDirFmt` with one sample per MAG."""
    root.mkdir()
    manifest = ["sample-id,mag-id,filename\n"]
    for i, mag_id in enumerate(mag_ids, start=1):
        contig_ids = None if contig_ids_by_mag is None else contig_ids_by_mag[mag_id]
        sample_id = f"sample{i}"
        sample_dir = root / sample_id
        sample_dir.mkdir()
        filename = f"{sample_id}/{mag_id}.fasta"
        (root / filename).write_text(_build_fasta_content(contig_ids))
        manifest.append(f"{sample_id},{mag_id},{filename}\n")
    (root / "MANIFEST").write_text("".join(manifest))
    return MultiMAGSequencesDirFmt(root, mode="r")


def _build_destination(root, destination_type, mag_ids, contig_ids_by_mag=None):
    """Builds a destination sequences fixture of the given `destination_type`
    ("feature_data" or "sample_data")."""
    if destination_type == "feature_data":
        return _build_mag_dirfmt(root, mag_ids, contig_ids_by_mag)
    return _build_multi_mag_dirfmt(root, mag_ids, contig_ids_by_mag)


class TestTransferAnnotations(TestPluginBase):
    package = "q2_annotate.eggnog.tests"

    def setUp(self):
        super().setUp()
        self.source_annotations = OrthologAnnotationDirFmt(
            self.get_data_path("annotations/"), mode="r"
        )
        self.destination_sequences = MAGSequencesDirFmt(
            self.get_data_path("mag-sequences-for-transfer/"), mode="r"
        )

    def _build_sample_data_destination(self, mag_ids):
        root = Path(self.temp_dir.name, "sample-destination")
        return _build_destination(root, "sample_data", mag_ids)

    def test_transfer_to_feature_data(self):
        result = _copy_mag_annotations(
            self.source_annotations, self.destination_sequences
        )
        src = self.source_annotations.annotation_dict()
        self.assertEqual(
            set(result.annotation_dict().keys()),
            {
                "1e9ffc02-0847-4f2c-b1e2-3965a4a78b15",
                "62e07985-2556-435c-9e02-e7f94b8df07d",
            },
        )
        for uuid, path in result.annotation_dict().items():
            self.assertTrue(filecmp.cmp(src[uuid], path, shallow=False))

    def test_transfer_to_sample_data(self):
        destination_sequences = self._build_sample_data_destination([MAG1, MAG2])
        result = _copy_mag_annotations(
            self.source_annotations, destination_sequences
        )
        src = self.source_annotations.annotation_dict()
        self.assertEqual(set(result.annotation_dict().keys()), {MAG1, MAG2})
        for uuid, path in result.annotation_dict().items():
            self.assertTrue(filecmp.cmp(src[uuid], path, shallow=False))

    def test_transfer_raises_on_no_match(self):
        uuid = "00000000-0000-4000-8000-000000000000"
        Path(self.temp_dir.name, f"{uuid}.fasta").touch()
        destination_sequences = MAGSequencesDirFmt(self.temp_dir.name, mode="r")
        with self.assertRaisesRegex(
            ValueError, "No annotation files matched the destination MAG IDs"
        ):
            _copy_mag_annotations(self.source_annotations, destination_sequences)

    def test_transfer_warns_on_partial_match(self):
        missing_mag = "00000000-0000-4000-8000-000000000000"
        root = Path(self.temp_dir.name, "partial-destination")
        root.mkdir()
        (root / f"{MAG1}.fasta").write_text(">placeholder\nACGT\n")
        (root / f"{missing_mag}.fasta").write_text(">placeholder\nACGT\n")
        destination_sequences = MAGSequencesDirFmt(root, mode="r")

        with self.assertWarns(UserWarning) as cm:
            result = _copy_mag_annotations(
                self.source_annotations, destination_sequences
            )
        self.assertIn("had no matching annotation file", str(cm.warning))
        self.assertIn(missing_mag, str(cm.warning))
        self.assertEqual(set(result.annotation_dict().keys()), {MAG1})


class TestAnnotateMagsFromContigs(TestPluginBase):
    package = "q2_annotate.eggnog.tests"

    def setUp(self):
        super().setUp()
        self.source_annotations = OrthologAnnotationDirFmt(
            self.get_data_path("contig-annotations/"), mode="r"
        )

        self.destination_sequences = _build_mag_dirfmt(
            Path(self.temp_dir.name, "initial-destination"), (CONTIG_MAG1, CONTIG_MAG2)
        )
        self.source_contig_map = MAGtoContigsDirFmt(
            self.get_data_path("mag-to-contigs/"), mode="r"
        ).file.view(dict)
        self.source_contig_map_partial = MAGtoContigsDirFmt(
            self.get_data_path("mag-to-contigs-partial/"), mode="r"
        ).file.view(dict)
        self.source_contig_map_nomatch = MAGtoContigsDirFmt(
            self.get_data_path("mag-to-contigs-nomatch/"), mode="r"
        ).file.view(dict)

    def _build_destination_sequences(
        self, destination_type, contig_ids_by_mag=None, root_name=None
    ):
        root = Path(self.temp_dir.name, root_name or f"{destination_type}-destination")
        return _build_destination(
            root, destination_type, (CONTIG_MAG1, CONTIG_MAG2), contig_ids_by_mag
        )

    def _query_ids(self, result, mag_uuid):
        df = pd.read_csv(result.annotation_dict()[mag_uuid], sep="\t", skiprows=4)
        return df[df.columns[0]].tolist()

    def test_resolve_contig_map(self):
        resolved_map = _resolve_contig_map(
            self.destination_sequences, self.source_contig_map
        )
        self.assertEqual(set(resolved_map.keys()), {CONTIG_MAG1, CONTIG_MAG2})

    def test_load_annotation_rows(self):
        obs = _load_annotation_rows(self.source_annotations)
        self.assertEqual(len(obs), 4)
        self.assertFalse(obs[obs.columns[0]].astype(str).str.startswith("##").any())

    def test_reverse_contig_map(self):
        source_contig_map = {
            MAG1: ["k141_100", "k141_200"],
            MAG2: ["k141_300"],
        }
        contig_to_mag, n_contigs_by_mag = _reverse_contig_map(source_contig_map)
        self.assertEqual(contig_to_mag["k141_100"], MAG1)
        self.assertEqual(contig_to_mag["k141_300"], MAG2)
        self.assertEqual(n_contigs_by_mag[MAG1], 2)
        self.assertEqual(n_contigs_by_mag[MAG2], 1)

    def test_map_rows_to_mag_ids(self):
        df = pd.DataFrame(
            {
                "#query": ["k141_100_0", "k141_200_1", "k141_999_0"],
                "seed_ortholog": ["a", "b", "c"],
            }
        )
        contig_to_mag = {"k141_100": MAG1, "k141_200": MAG2}
        obs = _map_rows_to_mag_ids(df, contig_to_mag)
        self.assertEqual(obs.loc[0, "mag_uuid"], MAG1)
        self.assertEqual(obs.loc[1, "mag_uuid"], MAG2)
        self.assertTrue(pd.isna(obs.loc[2, "mag_uuid"]))

    def test_require_matched_annotation_rows(self):
        df = pd.DataFrame(
            {
                "#query": ["q1", "q2", "q3"],
                "mag_uuid": [MAG1, None, MAG2],
            }
        )
        matched, total = _require_matched_annotation_rows(df)
        self.assertEqual(len(matched), 2)
        self.assertEqual(total, 3)

    def test_require_matched_annotation_rows_raises_on_empty(self):
        df = pd.DataFrame({"#query": ["q1"], "mag_uuid": [None]})
        with self.assertRaisesRegex(ValueError, "No annotation rows could be"):
            _require_matched_annotation_rows(df)

    def test_warn_unmatched_annotation_rows_warns_when_nonzero(self):
        with self.assertWarns(UserWarning):
            _warn_unmatched_annotation_rows(1, 4)

    def test_write_grouped_annotations(self):
        matched = pd.DataFrame(
            {
                "#query": ["k141_100_0", "k141_300_0"],
                "seed_ortholog": ["ortholog1", "ortholog2"],
                "mag_uuid": [MAG1, MAG2],
            }
        )
        result = _write_grouped_annotations(
            matched,
            {MAG1: 2, MAG2: 1},
        )
        lines = Path(result.annotation_dict()[MAG1]).read_text().splitlines()
        self.assertIn("## Source: contig-level annotations", lines[1])
        self.assertIn("#query\tseed_ortholog", lines[4])
        self.assertIn("k141_100_0\tortholog1", lines[5])

    def test_aggregate_contigs_into_mags(self):
        cases = {
            "feature_data_with_contig_map": (
                self.destination_sequences,
                self.source_contig_map,
            ),
            "sample_data_with_contig_map": (
                self._build_destination_sequences(
                    "sample_data", root_name="sample-data-with-contig-map"
                ),
                self.source_contig_map,
            ),
            "feature_data_without_contig_map": (
                self._build_destination_sequences(
                    "feature_data",
                    self.source_contig_map,
                    root_name="feature-data-without-contig-map",
                ),
                None,
            ),
            "sample_data_without_contig_map": (
                self._build_destination_sequences(
                    "sample_data",
                    self.source_contig_map,
                    root_name="sample-data-without-contig-map",
                ),
                None,
            ),
        }
        for name, (destination_sequences, contig_map) in cases.items():
            with self.subTest(name=name):
                result = _annotate_mags_from_contigs(
                    self.source_annotations,
                    destination_sequences,
                    contig_map,
                )
                self.assertEqual(
                    set(result.annotation_dict().keys()), {CONTIG_MAG1, CONTIG_MAG2}
                )
                self.assertEqual(
                    sorted(self._query_ids(result, CONTIG_MAG1)),
                    ["k141_100_0", "k141_100_1", "k141_200_0"],
                )
                self.assertEqual(self._query_ids(result, CONTIG_MAG2), ["k141_300_0"])

    def test_aggregate_warns_on_unmatched_rows(self):
        with self.assertWarns(UserWarning):
            result = _annotate_mags_from_contigs(
                self.source_annotations,
                self.destination_sequences,
                self.source_contig_map_partial,
            )
        self.assertEqual(set(result.annotation_dict().keys()), {CONTIG_MAG1})

    def test_aggregate_raises_when_nothing_matches(self):
        with self.assertRaisesRegex(ValueError, "No annotation rows could be"):
            _annotate_mags_from_contigs(
                self.source_annotations,
                self.destination_sequences,
                self.source_contig_map_nomatch,
            )


class TestTransferEggnogAnnotationsPipeline(TestPluginBase):
    package = "q2_annotate.eggnog.tests"

    def setUp(self):
        super().setUp()
        self.transfer_eggnog_annotations = self.plugin.pipelines[
            "transfer_eggnog_annotations"
        ]

    def _build_typed_source(self, data_dir, type_str):
        fmt = OrthologAnnotationDirFmt(self.get_data_path(data_dir), mode="r")
        return Artifact.import_data(type_str, fmt)

    def _build_mag_typed_source(self):
        return self._build_typed_source(
            "annotations/", "GenomeData[NOG % Properties('mags')]"
        )

    def _build_feature_data_mag_destination(self):
        fmt = MAGSequencesDirFmt(
            self.get_data_path("mag-sequences-for-transfer/"), mode="r"
        )
        return Artifact.import_data("FeatureData[MAG]", fmt)

    def _build_destination_artifact(self, destination_type, mag_ids, root_name=None):
        root = Path(self.temp_dir.name, root_name or f"{destination_type}-destination")
        fmt = _build_destination(root, destination_type, mag_ids)
        type_str = (
            "FeatureData[MAG]"
            if destination_type == "feature_data"
            else "SampleData[MAGs]"
        )
        return Artifact.import_data(type_str, fmt)

    def _build_contig_map_artifact(self):
        fmt = MAGtoContigsDirFmt(self.get_data_path("mag-to-contigs/"), mode="r")
        return Artifact.import_data("FeatureMap[MAGtoContigs]", fmt)

    def test_mag_typed_to_destination(self):
        destinations = {
            "feature_data": self._build_feature_data_mag_destination(),
            "sample_data": self._build_destination_artifact(
                "sample_data", [MAG1, MAG2]
            ),
        }
        for name, destination in destinations.items():
            with self.subTest(name=name):
                (result,) = self.transfer_eggnog_annotations(
                    self._build_mag_typed_source(), destination
                )
                self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
                obs = result.view(OrthologAnnotationDirFmt)
                self.assertEqual(set(obs.annotation_dict().keys()), {MAG1, MAG2})

    def test_mag_typed_rejects_contig_map(self):
        with self.assertRaisesRegex(
            Exception, "only valid for contig-level source annotations"
        ):
            self.transfer_eggnog_annotations(
                self._build_mag_typed_source(),
                self._build_feature_data_mag_destination(),
                self._build_contig_map_artifact(),
            )

    def test_contig_typed_to_destination_with_contig_map(self):
        destinations = {
            "feature_data": self._build_destination_artifact(
                "feature_data",
                (CONTIG_MAG1, CONTIG_MAG2),
                root_name="contig-feature-destination",
            ),
            "sample_data": self._build_destination_artifact(
                "sample_data",
                (CONTIG_MAG1, CONTIG_MAG2),
                root_name="contig-sample-destination",
            ),
        }
        for name, destination in destinations.items():
            with self.subTest(name=name):
                (result,) = self.transfer_eggnog_annotations(
                    self._build_typed_source(
                        "contig-annotations/",
                        "GenomeData[NOG % Properties('contigs')]",
                    ),
                    destination,
                    self._build_contig_map_artifact(),
                )
                obs = result.view(OrthologAnnotationDirFmt)
                self.assertEqual(
                    set(obs.annotation_dict().keys()),
                    {CONTIG_MAG1, CONTIG_MAG2},
                )
