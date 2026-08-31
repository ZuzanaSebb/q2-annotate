# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import filecmp
import re
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

    def test_build_contig_map_from_derep_mags(self):
        contig_map_from_derep_mags = _build_contig_map(self.destination_sequences)
        self.assertEqual(
            contig_map_from_derep_mags,
            self.expected_contig_map,
        )

    def test_build_contig_map_from_mags_per_sample(self):
        contig_map_from_mags_per_sample = _build_contig_map(
            self.destination_sequences_per_sample
        )
        self.assertEqual(
            contig_map_from_mags_per_sample,
            self.expected_contig_map,
        )

    def test_get_mag_ids_from_derep_mags(self):
        mag_ids_from_derep_mags = _get_mag_ids(self.destination_sequences)
        expected_mag_ids = {MAG1, MAG2}
        self.assertEqual(mag_ids_from_derep_mags, expected_mag_ids)

    def test_get_mag_ids_from_mags_per_sample(self):
        mag_ids_from_mags_per_sample = _get_mag_ids(
            self.destination_sequences_per_sample
        )
        expected_mag_ids = {MAG1, MAG2}
        self.assertEqual(mag_ids_from_mags_per_sample, expected_mag_ids)

    def test_resolve_contig_map_from_derep_mags(self):
        resolved_map_from_derep_mags = _resolve_contig_map(
            self.destination_sequences, self.source_contig_map
        )

        self.assertEqual(
            resolved_map_from_derep_mags,
            self.expected_contig_map,
        )
        self.assertNotIn(
            "33333333-3333-4333-8333-333333333333", resolved_map_from_derep_mags
        )

    def test_resolve_contig_map_from_mags_per_sample(self):
        resolved_map_from_mags_per_sample = _resolve_contig_map(
            self.destination_sequences_per_sample
        )

        self.assertEqual(
            resolved_map_from_mags_per_sample,
            self.expected_contig_map,
        )

    def test_load_annotation_rows(self):
        obs = _load_annotation_rows(self.source_annotations)
        exp = pd.read_csv(
            self.get_data_path(
                "expected-grouped-annotations/expected_annotation_rows.tsv"
            ),
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        pd.testing.assert_frame_equal(obs, exp)

    def test_reverse_contig_map(self):
        source_contig_map = {
            MAG1: ["mtjebimcR24S9DZ62TY6Fh", "NNqHkme8fLmessev7CnUMU"],
            MAG2: ["ipio8kS3aBF5G9Lw6XsSxj"],
        }
        contig_to_mag, n_contigs_by_mag = _reverse_contig_map(source_contig_map)
        self.assertDictEqual(
            contig_to_mag,
            {
                "mtjebimcR24S9DZ62TY6Fh": MAG1,
                "NNqHkme8fLmessev7CnUMU": MAG1,
                "ipio8kS3aBF5G9Lw6XsSxj": MAG2,
            },
        )
        self.assertDictEqual(n_contigs_by_mag, {MAG1: 2, MAG2: 1})

    def test_map_rows_to_mag_ids(self):
        obs = _map_rows_to_mag_ids(self.test_annotations_df, self.test_contig_to_mag)
        exp = pd.Series([MAG1, MAG2, None], name="mag_uuid")
        pd.testing.assert_series_equal(obs["mag_uuid"], exp)

    def test_require_matched_annotation_rows(self):
        tagged = self.test_annotations_df.assign(mag_uuid=[MAG1, MAG2, None])
        matched, _ = _require_matched_annotation_rows(tagged)
        expected = pd.DataFrame(
            {
                "#query": ["mtjebimcR24S9DZ62TY6Fh_0", "NNqHkme8fLmessev7CnUMU_1"],
                "seed_ortholog": ["a", "b"],
                "mag_uuid": [MAG1, MAG2],
            }
        )
        pd.testing.assert_frame_equal(matched.reset_index(drop=True), expected)

    def test_require_matched_annotation_rows_raises_on_empty(self):
        tagged = self.test_annotations_df.assign(mag_uuid=None)
        with self.assertRaisesRegex(ValueError, "No annotation rows could be matched"):
            _require_matched_annotation_rows(tagged)

    def test_warn_unmatched_annotation_rows_warns_when_nonzero(self):
        tagged = self.test_annotations_df.assign(mag_uuid=[MAG1, MAG2, None])
        matched, total = _require_matched_annotation_rows(tagged)
        unmatched = total - len(matched)
        expected_msg = (
            "1 of 3 annotation row(s) (33.3%) could not be matched to any MAG "
            "in the destination sequences and were skipped."
        )
        with self.assertWarnsRegex(UserWarning, re.escape(expected_msg)):
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
        exp_MAG1 = Path(
            self.get_data_path(
                f"expected-grouped-annotations/{MAG1}.emapper.annotations"
            )
        ).read_text()
        exp_MAG2 = Path(
            self.get_data_path(
                f"expected-grouped-annotations/{MAG2}.emapper.annotations"
            )
        ).read_text()

        self.assertEqual(Path(result.annotation_dict()[MAG1]).read_text(), exp_MAG1)
        self.assertEqual(Path(result.annotation_dict()[MAG2]).read_text(), exp_MAG2)

    def _assert_transfer(self, destination_sequences, source_contig_map=None):
        """Run _transfer_annotations_from_contigs and check that produced
        annotations are correctly aggregated for MAG1 and MAG2, and
        that annotations from an unrelated id got excluded."""

        with self.assertWarns(UserWarning):
            result = _transfer_annotations_from_contigs(
                self.source_annotations,
                destination_sequences,
                source_contig_map,
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

    def test_aggregate_contigs_into_mags_feature_data_with_contig_map(self):
        self._assert_transfer(self.destination_sequences, self.source_contig_map)

    def test_aggregate_contigs_into_mags_sample_data_with_contig_map(self):
        self._assert_transfer(
            self.destination_sequences_per_sample, self.source_contig_map
        )

    def test_aggregate_contigs_into_mags_feature_data_without_contig_map(self):
        self._assert_transfer(self.destination_sequences)

    def test_aggregate_contigs_into_mags_sample_data_without_contig_map(self):
        self._assert_transfer(self.destination_sequences_per_sample)

    def test_aggregate_raises_when_nothing_matches(self):
        with self.assertRaisesRegex(ValueError, "No annotation rows could be matched"):
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

        self.source_mags = Artifact.import_data(
            "GenomeData[NOG % Properties('mags')]",
            OrthologAnnotationDirFmt(self.get_data_path("annotations/"), mode="r"),
        )
        self.source_contigs = Artifact.import_data(
            "GenomeData[NOG % Properties('contigs')]",
            OrthologAnnotationDirFmt(
                self.get_data_path("contig-annotations/"), mode="r"
            ),
        )
        self.contig_map = Artifact.import_data(
            "FeatureMap[MAGtoContigs]",
            MAGtoContigsDirFmt(self.get_data_path("mag-to-contigs/"), mode="r"),
        )
        self.mag_sequences = Artifact.import_data(
            "FeatureData[MAG]",
            MAGSequencesDirFmt(
                self.get_data_path("mag-sequences-for-transfer/"), mode="r"
            ),
        )
        self.mag_sequences_per_sample = Artifact.import_data(
            "SampleData[MAGs]",
            MultiMAGSequencesDirFmt(
                self.get_data_path("mag-sequences-for-transfer-per-sample/"), mode="r"
            ),
        )

    def test_transfer_annotations_from_mags_to_mags_per_sample(self):
        (result,) = self.transfer_eggnog_annotations(
            self.source_mags, self.mag_sequences_per_sample
        )
        self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
        self.assertEqual(
            set(result.view(OrthologAnnotationDirFmt).annotation_dict().keys()),
            {MAG1, MAG2},
        )

        with self.assertWarnsRegex(
            UserWarning, "only valid for contig-level source annotations"
        ):
            (result,) = self.transfer_eggnog_annotations(
                self.source_mags, self.mag_sequences_per_sample, self.contig_map
            )
        self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
        self.assertEqual(
            set(result.view(OrthologAnnotationDirFmt).annotation_dict().keys()),
            {MAG1, MAG2},
        )

    def test_transfer_annotations_from_mags_to_derep_mags(self):
        (result,) = self.transfer_eggnog_annotations(
            self.source_mags, self.mag_sequences
        )
        self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
        self.assertEqual(
            set(result.view(OrthologAnnotationDirFmt).annotation_dict().keys()),
            {MAG1, MAG2},
        )

        with self.assertWarnsRegex(
            UserWarning, "only valid for contig-level source annotations"
        ):
            (result,) = self.transfer_eggnog_annotations(
                self.source_mags, self.mag_sequences, self.contig_map
            )
        self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
        self.assertEqual(
            set(result.view(OrthologAnnotationDirFmt).annotation_dict().keys()),
            {MAG1, MAG2},
        )

    def test_transfer_annotations_from_contigs_to_mags_per_sample(self):
        (result,) = self.transfer_eggnog_annotations(
            self.source_contigs, self.mag_sequences_per_sample, self.contig_map
        )
        self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
        self.assertEqual(
            set(result.view(OrthologAnnotationDirFmt).annotation_dict().keys()),
            {MAG1, MAG2},
        )

    def test_transfer_annotations_from_contigs_to_derep_mags(self):
        (result,) = self.transfer_eggnog_annotations(
            self.source_contigs, self.mag_sequences, self.contig_map
        )
        self.assertTrue(result.type <= GenomeData[NOG % Properties("mags")])
        self.assertEqual(
            set(result.view(OrthologAnnotationDirFmt).annotation_dict().keys()),
            {MAG1, MAG2},
        )
