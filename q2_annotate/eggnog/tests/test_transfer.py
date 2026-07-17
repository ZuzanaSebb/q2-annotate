# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team.
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
    _transfer_annotations_from_contigs,
    _transfer_annotations_from_mags,
    _load_annotation_rows,
    _map_rows_to_mag_ids,
    _require_matched_annotation_rows,
    _resolve_contig_map,
    _reverse_contig_map,
    _warn_unmatched_annotation_rows,
    _write_grouped_annotations,
    _build_contig_map,
    _get_mag_ids,
)
from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.feature_map import MAGtoContigsDirFmt
from q2_types.genome_data import GenomeData, NOG, OrthologAnnotationDirFmt
from q2_types.per_sample_sequences import MultiMAGSequencesDirFmt

MAG1 = "1e9ffc02-0847-4f2c-b1e2-3965a4a78b15"
MAG2 = "62e07985-2556-435c-9e02-e7f94b8df07d"


class TestTransferAnnotationsFromMags(TestPluginBase):
    package = "q2_annotate.eggnog.tests"

    # reusing annotations dir storing per-MAG annotations
    def setUp(self):
        super().setUp()
        self.source_annotations = OrthologAnnotationDirFmt(
            self.get_data_path("annotations/"), mode="r"
        )

    def test_transfer_to_feature_data(self):
        destination_sequences = MAGSequencesDirFmt(
            self.get_data_path("mag-sequences-for-transfer/"), mode="r"
        )
        result = _transfer_annotations_from_mags(
            self.source_annotations, destination_sequences
        )
        src = self.source_annotations.annotation_dict()
        self.assertEqual(set(result.annotation_dict().keys()), {MAG1, MAG2})
        for uuid, path in result.annotation_dict().items():
            self.assertTrue(filecmp.cmp(src[uuid], path, shallow=False))

    def test_transfer_to_sample_data(self):
        destination_sequences = MultiMAGSequencesDirFmt(
            self.get_data_path("mag-sequences-for-transfer-per-sample/"), mode="r"
        )
        result = _transfer_annotations_from_mags(
            self.source_annotations, destination_sequences
        )
        src = self.source_annotations.annotation_dict()
        self.assertEqual(set(result.annotation_dict().keys()), {MAG1, MAG2})
        for uuid, path in result.annotation_dict().items():
            self.assertTrue(filecmp.cmp(src[uuid], path, shallow=False))

    def test_transfer_raises_on_no_match(self):
        destination_sequences = MAGSequencesDirFmt(
            self.get_data_path("mag-sequences-unmatched/"), mode="r"
        )
        with self.assertRaisesRegex(
            ValueError, "No annotation files matched the destination MAG IDs"
        ):
            _transfer_annotations_from_mags(
                self.source_annotations, destination_sequences
            )

    def test_transfer_warns_on_partial_match(self):
        destination_sequences = MAGSequencesDirFmt(
            self.get_data_path("mag-sequences-partial-match/"), mode="r"
        )
        with self.assertWarns(UserWarning) as cm:
            result = _transfer_annotations_from_mags(
                self.source_annotations, destination_sequences
            )
        self.assertIn("had no matching annotation file", str(cm.warning))
        self.assertIn("00000000-0000-4000-8000-000000000000", str(cm.warning))
        self.assertEqual(set(result.annotation_dict().keys()), {MAG1})


class TestTransferAnnotationsFromContigs(TestPluginBase):
    package = "q2_annotate.eggnog.tests"

    def setUp(self):
        super().setUp()
        self.source_annotations = OrthologAnnotationDirFmt(
            self.get_data_path("contig-annotations/"), mode="r"
        )
        self.destination_sequences = MAGSequencesDirFmt(
            self.get_data_path("mag-sequences-for-transfer/"), mode="r"
        )
        self.destination_sequences_per_sample = MultiMAGSequencesDirFmt(
            self.get_data_path("mag-sequences-for-transfer-per-sample/"), mode="r"
        )
        self.source_contig_map = MAGtoContigsDirFmt(
            self.get_data_path("mag-to-contigs/"), mode="r"
        ).file.view(dict)
        self.source_contig_map_nomatch = MAGtoContigsDirFmt(
            self.get_data_path("mag-to-contigs-nomatch/"), mode="r"
        ).file.view(dict)
        self.test_annotations_df = pd.DataFrame(
            {
                "#query": [
                    "mtjebimcR24S9DZ62TY6Fh_0",
                    "NNqHkme8fLmessev7CnUMU_1",
                    "ipio8kS3aBF5G9Lw6XsSxj_0",
                ],
                "seed_ortholog": ["a", "b", "c"],
            }
        )
        self.test_contig_to_mag = {
            "mtjebimcR24S9DZ62TY6Fh": MAG1,
            "NNqHkme8fLmessev7CnUMU": MAG2,
        }
        self.expected_contig_map = {
            MAG1: ["mtjebimcR24S9DZ62TY6Fh", "NNqHkme8fLmessev7CnUMU"],
            MAG2: ["ipio8kS3aBF5G9Lw6XsSxj"],
        }

    def _query_ids(self, result, mag_uuid):
        df = pd.read_csv(result.annotation_dict()[mag_uuid], sep="\t", skiprows=4)
        return df[df.columns[0]].tolist()

    def test_build_contig_map(self):
        contig_map_from_derp_mags = _build_contig_map(self.destination_sequences)
        contig_map_from_mags_per_sample = _build_contig_map(
            self.destination_sequences_per_sample
        )

        self.assertEqual(
            contig_map_from_derp_mags,
            self.expected_contig_map,
        )
        self.assertEqual(
            contig_map_from_mags_per_sample,
            self.expected_contig_map,
        )

    def test_get_mag_ids(self):
        mag_ids_from_derp_mags = _get_mag_ids(self.destination_sequences)
        mag_ids_from_mags_per_sample = _get_mag_ids(
            self.destination_sequences_per_sample
        )
        expected_mag_ids = {MAG1, MAG2}
        self.assertEqual(mag_ids_from_derp_mags, expected_mag_ids)
        self.assertEqual(mag_ids_from_mags_per_sample, expected_mag_ids)

    def test_resolve_contig_map(self):
        resolved_map_from_derp_mags = _resolve_contig_map(
            self.destination_sequences, self.source_contig_map
        )
        resolved_map_from_mags_per_sample = _resolve_contig_map(
            self.destination_sequences_per_sample
        )
        self.assertEqual(
            resolved_map_from_derp_mags,
            self.expected_contig_map,
        )
        self.assertEqual(
            resolved_map_from_mags_per_sample,
            self.expected_contig_map,
        )
        self.assertNotIn(
            "33333333-3333-4333-8333-333333333333", resolved_map_from_derp_mags
        )

    def test_load_annotation_rows(self):
        obs = _load_annotation_rows(self.source_annotations)
        self.assertEqual(len(obs), 10)
        self.assertFalse(obs[obs.columns[0]].astype(str).str.startswith("##").any())
        self.assertIn("pXaG7nQ3mZtY8LbKdWs1Rf_1", obs[obs.columns[0]].tolist())

    def test_reverse_contig_map(self):
        source_contig_map = {
            MAG1: ["mtjebimcR24S9DZ62TY6Fh", "NNqHkme8fLmessev7CnUMU"],
            MAG2: ["ipio8kS3aBF5G9Lw6XsSxj"],
        }
        contig_to_mag, n_contigs_by_mag = _reverse_contig_map(source_contig_map)
        self.assertEqual(contig_to_mag["mtjebimcR24S9DZ62TY6Fh"], MAG1)
        self.assertEqual(contig_to_mag["ipio8kS3aBF5G9Lw6XsSxj"], MAG2)
        self.assertEqual(n_contigs_by_mag[MAG1], 2)
        self.assertEqual(n_contigs_by_mag[MAG2], 1)

    def test_map_rows_to_mag_ids(self):
        obs = _map_rows_to_mag_ids(self.test_annotations_df, self.test_contig_to_mag)
        self.assertEqual(obs.loc[0, "mag_uuid"], MAG1)
        self.assertEqual(obs.loc[1, "mag_uuid"], MAG2)
        self.assertTrue(pd.isna(obs.loc[2, "mag_uuid"]))

    def test_require_matched_annotation_rows(self):
        tagged = self.test_annotations_df.assign(mag_uuid=[MAG1, MAG2, None])
        matched, total = _require_matched_annotation_rows(tagged)
        self.assertEqual(len(matched), 2)
        self.assertEqual(total, 3)

    def test_require_matched_annotation_rows_raises_on_empty(self):
        tagged = self.test_annotations_df.assign(mag_uuid=None)
        with self.assertRaisesRegex(ValueError, "No annotation rows could be"):
            _require_matched_annotation_rows(tagged)

    def test_warn_unmatched_annotation_rows_warns_when_nonzero(self):
        tagged = self.test_annotations_df.assign(mag_uuid=[MAG1, MAG2, None])
        matched, total = _require_matched_annotation_rows(tagged)
        unmatched = total - len(matched)
        with self.assertWarns(UserWarning):
            _warn_unmatched_annotation_rows(unmatched, total)

    def test_write_grouped_annotations(self):
        matched = pd.DataFrame(
            {
                "#query": ["mtjebimcR24S9DZ62TY6Fh_0", "ipio8kS3aBF5G9Lw6XsSxj_0"],
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
        self.assertIn("mtjebimcR24S9DZ62TY6Fh_0\tortholog1", lines[5])

    def test_aggregate_contigs_into_mags(self):
        cases = {
            "feature_data_with_contig_map": (
                self.destination_sequences,
                self.source_contig_map,
            ),
            "sample_data_with_contig_map": (
                MultiMAGSequencesDirFmt(
                    self.get_data_path("mag-sequences-for-transfer-per-sample/"),
                    mode="r",
                ),
                self.source_contig_map,
            ),
            "feature_data_without_contig_map": (
                self.destination_sequences,
                None,
            ),
            "sample_data_without_contig_map": (
                MultiMAGSequencesDirFmt(
                    self.get_data_path("mag-sequences-for-transfer-per-sample/"),
                    mode="r",
                ),
                None,
            ),
        }
        for name, (destination_sequences, contig_map) in cases.items():
            with self.subTest(name=name):
                with self.assertWarns(UserWarning):
                    result = _transfer_annotations_from_contigs(
                        self.source_annotations,
                        destination_sequences,
                        contig_map,
                    )
                self.assertEqual(set(result.annotation_dict().keys()), {MAG1, MAG2})
                mag1_ids = sorted(self._query_ids(result, MAG1))
                mag2_ids = sorted(self._query_ids(result, MAG2))
                self.assertEqual(
                    mag1_ids,
                    [
                        "NNqHkme8fLmessev7CnUMU_1",
                        "NNqHkme8fLmessev7CnUMU_2",
                        "mtjebimcR24S9DZ62TY6Fh_1",
                        "mtjebimcR24S9DZ62TY6Fh_2",
                        "mtjebimcR24S9DZ62TY6Fh_3",
                        "mtjebimcR24S9DZ62TY6Fh_4",
                    ],
                )
                self.assertEqual(
                    mag2_ids,
                    [
                        "ipio8kS3aBF5G9Lw6XsSxj_1",
                        "ipio8kS3aBF5G9Lw6XsSxj_2",
                        "ipio8kS3aBF5G9Lw6XsSxj_3",
                    ],
                )
                self.assertNotIn("pXaG7nQ3mZtY8LbKdWs1Rf_1", mag1_ids + mag2_ids)

    def test_aggregate_raises_when_nothing_matches(self):
        with self.assertRaisesRegex(ValueError, "No annotation rows could be"):
            _transfer_annotations_from_contigs(
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

    def test_transfer_annotations_from_mags(self):
        annotations_dir = self.get_data_path("annotations/")
        annotations = OrthologAnnotationDirFmt(annotations_dir, mode="r")
        source = Artifact.import_data(
            "GenomeData[NOG % Properties('mags')]", annotations
        )

        mag_sequences_dir = self.get_data_path("mag-sequences-for-transfer/")
        mag_sequences = MAGSequencesDirFmt(mag_sequences_dir, mode="r")
        mag_sequences_artifact = Artifact.import_data("FeatureData[MAG]", mag_sequences)

        mag_sequences_per_sample_dir = self.get_data_path(
            "mag-sequences-for-transfer-per-sample/"
        )
        mag_sequences_per_sample = MultiMAGSequencesDirFmt(
            mag_sequences_per_sample_dir, mode="r"
        )
        mag_sequences_per_sample_artifact = Artifact.import_data(
            "SampleData[MAGs]", mag_sequences_per_sample
        )

        destinations = {
            "mag-sequences-for-transfer": mag_sequences_artifact,
            "mag-sequences-for-transfer-per-sample": mag_sequences_per_sample_artifact,
        }
        for name, destination in destinations.items():
            with self.subTest(name=name):
                (result,) = self.transfer_eggnog_annotations(source, destination)
                self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
                obs = result.view(OrthologAnnotationDirFmt)
                self.assertEqual(set(obs.annotation_dict().keys()), {MAG1, MAG2})

        contig_map_dir = self.get_data_path("mag-to-contigs/")
        contig_map = MAGtoContigsDirFmt(contig_map_dir, mode="r")
        contig_map = Artifact.import_data("FeatureMap[MAGtoContigs]", contig_map)

        with self.assertWarnsRegex(
            UserWarning, "only valid for contig-level source annotations"
        ):
            (result,) = self.transfer_eggnog_annotations(
                source, mag_sequences_artifact, contig_map
            )
        self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
        obs = result.view(OrthologAnnotationDirFmt)
        self.assertEqual(set(obs.annotation_dict().keys()), {MAG1, MAG2})

    def test_transfer_annotations_from_contigs(self):
        contig_annotations_dir = self.get_data_path("contig-annotations/")
        contig_annotations = OrthologAnnotationDirFmt(contig_annotations_dir, mode="r")
        source = Artifact.import_data(
            "GenomeData[NOG % Properties('contigs')]", contig_annotations
        )

        contig_map_dir = self.get_data_path("mag-to-contigs/")
        contig_map = MAGtoContigsDirFmt(contig_map_dir, mode="r")
        contig_map = Artifact.import_data("FeatureMap[MAGtoContigs]", contig_map)

        mag_sequences_dir = self.get_data_path("mag-sequences-for-transfer/")
        mag_sequences = MAGSequencesDirFmt(mag_sequences_dir, mode="r")
        mag_sequences_artifact = Artifact.import_data("FeatureData[MAG]", mag_sequences)

        mag_sequences_per_sample_dir = self.get_data_path(
            "mag-sequences-for-transfer-per-sample/"
        )
        mag_sequences_per_sample = MultiMAGSequencesDirFmt(
            mag_sequences_per_sample_dir, mode="r"
        )
        mag_sequences_per_sample_artifact = Artifact.import_data(
            "SampleData[MAGs]", mag_sequences_per_sample
        )

        destinations = {
            "mag-sequences-for-transfer": mag_sequences_artifact,
            "mag-sequences-for-transfer-per-sample": mag_sequences_per_sample_artifact,
        }
        for name, destination in destinations.items():
            with self.subTest(name=name):
                (result,) = self.transfer_eggnog_annotations(
                    source, destination, contig_map
                )
                obs = result.view(OrthologAnnotationDirFmt)
                self.assertEqual(
                    set(obs.annotation_dict().keys()),
                    {MAG1, MAG2},
                )
