"""Tests for per-record label loading."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ecgbench.config import LabelConfig
from ecgbench.labels import (
    LabelSourceMissingError,
    LabelsUnavailableError,
    load_labels,
)


def _with_labels(config, **kwargs):
    """Attach a LabelConfig to a config fixture."""
    spec = LabelConfig(
        source_csv="source_labels.csv",
        join_column="rec",
        **kwargs,
    )
    return replace(config, labels=spec)


class TestDeclarativeLoader:
    def test_reads_source_csv(self, sample_config, tmp_labels_data):
        config = _with_labels(sample_config)
        df = load_labels(config, data_path=tmp_labels_data)

        assert len(df) == 3
        # Indexed by the dataset's record ID column, whatever the source calls it.
        assert df.index.name == config.record_id_column
        assert list(df.index) == ["rec_0", "rec_1", "rec_2"]
        # Join column is consumed by the index, not duplicated as a column.
        assert "rec" not in df.columns
        assert df.loc["rec_2", "diagnosis"] == "AFIB"

    def test_column_subset(self, sample_config, tmp_labels_data):
        config = _with_labels(sample_config, columns=["diagnosis", "age"])
        df = load_labels(config, data_path=tmp_labels_data)

        assert list(df.columns) == ["diagnosis", "age"]

    def test_all_columns_when_unspecified(self, sample_config, tmp_labels_data):
        config = _with_labels(sample_config, columns=None)
        df = load_labels(config, data_path=tmp_labels_data)

        assert list(df.columns) == ["diagnosis", "age", "note"]

    def test_unknown_column_raises(self, sample_config, tmp_labels_data):
        config = _with_labels(sample_config, columns=["diagnosis", "nope"])

        with pytest.raises(ValueError, match=r"\['nope'\] not in"):
            load_labels(config, data_path=tmp_labels_data)

    def test_unknown_join_column_raises(self, sample_config, tmp_labels_data):
        config = replace(
            sample_config,
            labels=LabelConfig(source_csv="source_labels.csv", join_column="missing"),
        )

        with pytest.raises(ValueError, match="Join column 'missing' not in"):
            load_labels(config, data_path=tmp_labels_data)

    def test_missing_source_file_names_it(self, sample_config, tmp_path):
        config = _with_labels(sample_config)

        with pytest.raises(LabelSourceMissingError) as excinfo:
            load_labels(config, data_path=tmp_path)
        # The message must name the file and say labels are not on the Hub.
        assert "source_labels.csv" in str(excinfo.value)
        assert "fold CSVs only" in str(excinfo.value)

    def test_duplicate_record_ids_raise(self, sample_config, tmp_labels_data):
        pd.DataFrame({"rec": ["a", "a"], "diagnosis": ["X", "Y"]}).to_csv(
            tmp_labels_data / "source_labels.csv", index=False
        )
        config = _with_labels(sample_config)

        with pytest.raises(ValueError, match="duplicate record IDs"):
            load_labels(config, data_path=tmp_labels_data)


class TestUnavailableLabels:
    def test_no_labels_block(self, sample_config, tmp_path):
        with pytest.raises(LabelsUnavailableError, match="no labels block"):
            load_labels(sample_config, data_path=tmp_path)

    def test_explicitly_unavailable_gives_the_reason(self, sample_config, tmp_path):
        config = replace(
            sample_config,
            labels=LabelConfig(available=False, unavailable_reason="Only in the full release."),
        )

        with pytest.raises(LabelsUnavailableError, match="Only in the full release"):
            load_labels(config, data_path=tmp_path)


class TestPTBXLLoader:
    """PTB-XL derives superclasses from the shipped statement table."""

    def _load(self, ptbxl_config, tmp_ptbxl_label_data):
        config = replace(
            ptbxl_config,
            labels=LabelConfig(source_csv="ptbxl_database.csv", join_column="ecg_id"),
        )
        return load_labels(config, data_path=tmp_ptbxl_label_data)

    def test_derived_columns(self, ptbxl_config, tmp_ptbxl_label_data):
        df = self._load(ptbxl_config, tmp_ptbxl_label_data)

        assert df.index.name == "ecg_id"
        for col in ("scp_codes", "diagnostic_codes", "form_codes", "rhythm_codes",
                    "superclasses", "subclasses", "primary_superclass"):
            assert col in df.columns
        # Passthrough columns come along too.
        assert df.loc[1, "report"] == "sinusrhythmus"

    def test_scp_codes_parsed_to_dict(self, ptbxl_config, tmp_ptbxl_label_data):
        df = self._load(ptbxl_config, tmp_ptbxl_label_data)

        assert df.loc[1, "scp_codes"] == {"NORM": 100.0, "SR": 0.0}

    def test_multi_label_superclasses(self, ptbxl_config, tmp_ptbxl_label_data):
        df = self._load(ptbxl_config, tmp_ptbxl_label_data)

        assert df.loc[1, "superclasses"] == ["NORM"]
        # IMI -> MI and NDT -> STTC, so this record carries two.
        assert df.loc[2, "superclasses"] == ["MI", "STTC"]
        assert df.loc[2, "subclasses"] == ["IMI", "STTC"]

    def test_only_diagnostic_statements_count(self, ptbxl_config, tmp_ptbxl_label_data):
        """SR and ABQRS are rhythm/form statements, so record 3 has no superclass."""
        df = self._load(ptbxl_config, tmp_ptbxl_label_data)

        assert df.loc[3, "superclasses"] == []
        assert df.loc[3, "primary_superclass"] == "OTHER"
        assert df.loc[3, "rhythm_codes"] == ["SR"]
        assert df.loc[3, "form_codes"] == ["ABQRS"]

    def test_primary_superclass_is_deterministic_on_ties(
        self, ptbxl_config, tmp_ptbxl_label_data
    ):
        """Record 4 ties HYP against MI; the winner must not depend on dict order."""
        first = self._load(ptbxl_config, tmp_ptbxl_label_data)
        second = self._load(ptbxl_config, tmp_ptbxl_label_data)

        assert first.loc[4, "superclasses"] == ["HYP", "MI"]
        assert first.loc[4, "primary_superclass"] == second.loc[4, "primary_superclass"]
        # Ties break on the fixed SUPERCLASSES order, where MI precedes HYP.
        assert first.loc[4, "primary_superclass"] == "MI"

    def test_missing_statement_table_names_it(self, ptbxl_config, tmp_ptbxl_label_data):
        (tmp_ptbxl_label_data / "scp_statements.csv").unlink()

        with pytest.raises(LabelSourceMissingError, match="scp_statements.csv"):
            self._load(ptbxl_config, tmp_ptbxl_label_data)


class TestMultiHot:
    def test_encodes_in_fixed_class_order(self):
        from ecgbench.labels.ptbxl import SUPERCLASSES, multi_hot

        out = multi_hot([["NORM"], ["MI", "STTC"], []])

        assert out.shape == (3, len(SUPERCLASSES))
        assert out[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
        assert out[1].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
        assert out[2].tolist() == [0.0] * 5

    def test_unknown_class_is_ignored(self):
        from ecgbench.labels.ptbxl import multi_hot

        assert multi_hot([["NOT_A_CLASS"]]).sum() == 0.0

    def test_custom_class_list(self):
        from ecgbench.labels.ptbxl import multi_hot

        out = multi_hot([["AFIB"], ["SR"]], classes=["SR", "AFIB"])
        assert out[0].tolist() == [0.0, 1.0]
        assert out[1].tolist() == [1.0, 0.0]


class TestShippedConfigs:
    """The labels blocks in the shipped YAML must parse into something usable."""

    @pytest.mark.parametrize(
        ("slug", "source_csv", "join_column"),
        [
            ("ptbxl", "ptbxl_database.csv", "ecg_id"),
            ("chapman_shaoxing", "Diagnostics.csv", "FileName"),
            ("ecg_arrhythmia", "ecgbench_metadata.csv", "record_name"),
            ("ludb", "ludb.csv", "ID"),
        ],
    )
    def test_available_datasets_declare_a_source(self, slug, source_csv, join_column):
        from ecgbench.config import load_config

        spec = load_config(slug).labels
        assert spec is not None and spec.available
        assert spec.source_csv == source_csv
        assert spec.join_column == join_column

    def test_ptbdb_labels_come_from_headers_not_a_csv(self):
        """PTBDB ships no metadata file at all — the loader parses .hea comments."""
        from ecgbench.config import load_config

        spec = load_config("ptbdb").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # nothing to point at; headers are the source
        assert spec.join_column == "record_name"

    def test_variable_length_dataset_disables_the_truncation_check(self):
        """PTBDB records differ in length, so expected_samples must stay empty.

        check_truncated_signal returns [] when the rate has no entry; adding one
        would fail every record shorter than it.
        """
        from ecgbench.config import load_config

        assert load_config("ptbdb").validation.expected_samples == {}

    def test_mimic_demo_declares_labels_unavailable_with_a_reason(self):
        from ecgbench.config import load_config

        spec = load_config("mimic_iv_ecg_demo").labels
        assert spec is not None
        assert spec.available is False
        # An unavailable block is only useful if it says where to look instead.
        assert "MIMIC-IV" in spec.unavailable_reason


class TestChallenge2021Labels:
    """The packaged SNOMED table and the multi-label derivation over it."""

    def test_labels_come_from_headers_not_a_csv(self):
        """No metadata file ships, so labels do not depend on a prior pipeline run."""
        from ecgbench.config import load_config

        spec = load_config("challenge2021").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # the headers are the source
        assert spec.join_column == "record_name"

    def test_packaged_mapping_covers_the_challenge_classes(self):
        """133 codes, 30 scored, no duplicate code or abbreviation.

        No code table ships with the dataset, so this file is the only mapping
        from SNOMED codes to names. A duplicate code would silently drop a class.
        """
        from ecgbench.labels.challenge2021 import load_dx_mapping

        mapping = load_dx_mapping()
        assert len(mapping) == 133
        assert mapping.index.is_unique
        assert mapping["abbreviation"].is_unique
        assert int(mapping["scored"].sum()) == 30
        # Spot-check both halves of the table.
        assert mapping.loc["164889003", "abbreviation"] == "AF"
        assert bool(mapping.loc["164889003", "scored"]) is True
        assert bool(mapping.loc["164951009", "scored"]) is False  # abQRS, unscored

    def test_unknown_codes_are_kept_rather_than_dropped(self):
        """A code absent from the table must surface, not vanish silently."""
        import pandas as pd

        from ecgbench.labels.challenge2021 import UNMAPPED, attach_dx_columns

        # Two AF records make AF the common code, so the unknown one is rarest
        # and reaches the stratification label — where it must be visible as
        # UNMAPPED rather than passed off as a real class.
        df = attach_dx_columns(
            pd.DataFrame({"dx": ["164889003,999999999", "164889003", "164889003"]})
        )

        assert df.loc[0, "n_dx"] == 2
        assert df.loc[0, "dx_abbreviations"] == f"AF,{UNMAPPED}"
        assert df.loc[0, "scored_dx"] == "AF"  # unknown codes are never "scored"
        assert df.loc[0, "stratify_dx"] == "999999999"
        assert df.loc[0, "stratify_dx_abbreviation"] == UNMAPPED

    def test_scored_subset_excludes_unscored_codes(self):
        import pandas as pd

        from ecgbench.labels.challenge2021 import attach_dx_columns

        # AF is scored; abQRS (164951009) is not.
        df = attach_dx_columns(pd.DataFrame({"dx": ["164889003,164951009"]}))

        assert df.loc[0, "n_dx"] == 2
        assert df.loc[0, "scored_dx"] == "AF"
        assert df.loc[0, "n_scored_dx"] == 1

    def test_stratify_reduction_breaks_ties_deterministically(self):
        """Equally rare codes must resolve to the lowest numeric code, not scan order."""
        import pandas as pd

        from ecgbench.labels.challenge2021 import attach_dx_columns

        forward = attach_dx_columns(pd.DataFrame({"dx": ["164889003,164890007"]}))
        reversed_ = attach_dx_columns(pd.DataFrame({"dx": ["164890007,164889003"]}))

        assert forward.loc[0, "stratify_dx"] == "164889003"
        assert reversed_.loc[0, "stratify_dx"] == "164889003"


class TestChallenge2020Labels:
    """The packaged 2020 SNOMED table, and the duplicate-code defect it exposes."""

    def test_labels_come_from_headers_not_a_csv(self):
        """No metadata file ships, so labels do not depend on a prior pipeline run."""
        from ecgbench.config import load_config

        spec = load_config("challenge2020").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # the headers are the source
        assert spec.join_column == "record_name"

    def test_packaged_mapping_covers_the_challenge_classes(self):
        """111 codes, 27 scored, no duplicate code or abbreviation.

        No code table ships with the dataset, so this file is the only mapping
        from SNOMED codes to names. A duplicate code would silently drop a class.
        """
        from ecgbench.labels.challenge2020 import load_dx_mapping

        mapping = load_dx_mapping()
        assert len(mapping) == 111
        assert mapping.index.is_unique
        assert mapping["abbreviation"].is_unique
        assert int(mapping["scored"].sum()) == 27
        # Spot-check both halves of the table.
        assert mapping.loc["164889003", "abbreviation"] == "AF"
        assert bool(mapping.loc["164889003", "scored"]) is True
        assert bool(mapping.loc["164951009", "scored"]) is False  # abQRS, unscored

    def test_scored_subset_is_2020s_not_2021s(self):
        """The two challenges scored different class sets — do not share a table.

        2021's 30 scored classes are 2020's 27 plus three: PRWP and CLBBB, which
        do not occur in the 2020 release at all, and BBB, which does (137
        records) but was unscored that year. Nothing was scored in 2020 and
        dropped in 2021. Loading the wrong table silently changes the task.
        """
        from ecgbench.labels.challenge2020 import load_dx_mapping as load_2020
        from ecgbench.labels.challenge2021 import load_dx_mapping as load_2021

        m2020, m2021 = load_2020(), load_2021()
        scored_2020 = set(m2020.index[m2020["scored"]])
        scored_2021 = set(m2021.index[m2021["scored"]])

        assert len(scored_2020) == 27
        assert len(scored_2021) == 30
        assert not scored_2020 - scored_2021
        # PRWP, BBB, CLBBB
        assert scored_2021 - scored_2020 == {"365413008", "6374002", "733534002"}
        # BBB is in the 2020 table, just not scored there.
        assert bool(m2020.loc["6374002", "scored"]) is False
        # Every 2020 code is in the (larger) 2021 table, but not vice versa.
        assert set(m2020.index) < set(m2021.index)

    def test_repeated_codes_in_one_dx_field_are_deduplicated(self):
        """631 shipped records list a code twice; counting entries breaks the totals.

        This is the defect that makes v1.0.2 look as though it disagrees with the
        official dx_mapping table. 284470004 (PAC) is the code that does it.
        """
        import pandas as pd

        from ecgbench.labels.challenge2020 import attach_dx_columns

        df = attach_dx_columns(
            pd.DataFrame({"dx": ["251146004,284470004,284470004", "284470004,284470004,284470004"]})
        )

        assert df.loc[0, "dx"] == "251146004,284470004"
        assert df.loc[0, "n_dx"] == 2
        assert df.loc[0, "dx_abbreviations"] == "LQRSV,PAC"
        # A code repeated three times collapses to one, and does not become
        # "the rarest code" by virtue of being counted three times.
        assert df.loc[1, "dx"] == "284470004"
        assert df.loc[1, "n_dx"] == 1

    def test_deduplication_preserves_first_occurrence_order(self):
        """#Dx order is not meaningful, but it must at least be stable."""
        import pandas as pd

        from ecgbench.labels.challenge2020 import attach_dx_columns

        df = attach_dx_columns(pd.DataFrame({"dx": ["427084000,111975006,427084000"]}))

        assert df.loc[0, "dx"] == "427084000,111975006"

    def test_unknown_codes_are_kept_rather_than_dropped(self):
        """A code absent from the table must surface, not vanish silently."""
        import pandas as pd

        from ecgbench.labels.challenge2020 import UNMAPPED, attach_dx_columns

        df = attach_dx_columns(
            pd.DataFrame({"dx": ["164889003,999999999", "164889003", "164889003"]})
        )

        assert df.loc[0, "n_dx"] == 2
        assert df.loc[0, "dx_abbreviations"] == f"AF,{UNMAPPED}"
        assert df.loc[0, "scored_dx"] == "AF"  # unknown codes are never "scored"
        assert df.loc[0, "stratify_dx"] == "999999999"
        assert df.loc[0, "stratify_dx_abbreviation"] == UNMAPPED

    def test_stratify_reduction_breaks_ties_deterministically(self):
        """Equally rare codes must resolve to the lowest numeric code, not scan order."""
        import pandas as pd

        from ecgbench.labels.challenge2020 import attach_dx_columns

        forward = attach_dx_columns(pd.DataFrame({"dx": ["164889003,164890007"]}))
        reversed_ = attach_dx_columns(pd.DataFrame({"dx": ["164890007,164889003"]}))

        assert forward.loc[0, "stratify_dx"] == "164889003"
        assert reversed_.loc[0, "stratify_dx"] == "164889003"

    def test_age_sentinels_are_documented_and_not_silently_dropped(self):
        """300 means 'over 89' and -1 means nothing; both must stay distinguishable
        from a genuinely absent age, so the loader keeps all three states."""
        from ecgbench.labels.challenge2020 import AGE_SENTINELS

        assert AGE_SENTINELS == ("-1", "300")


class TestINCARTDBLabels:
    """Three comment lines per header, one of which is optional in half the records."""

    HEADER_WITH_DX = (
        "I01 12 257 462600\n"
        "I01.dat 16 306 16 0 1161 -11409 0 I\n"
        "#<age>: 65 <sex>: F <diagnoses> Coronary artery disease, arterial hypertension\n"
        "# patient 1\n"
        "# PVCs, noise\n"
    )
    # 34 of the 75 real records look like this: the <diagnoses> TOKEN is absent,
    # not merely empty. Note the trailing space, which the real files also have.
    HEADER_NO_DX = (
        "I08 12 257 462600\n"
        "I08.dat 16 612 16 0 0 0 0 I\n"
        "#<age>: 51 <sex>: F \n"
        "# patient 4\n"
        "# ventricular bigeminy, ventricular couplets\n"
    )

    def test_labels_come_from_headers_not_a_csv(self):
        from ecgbench.config import load_config

        spec = load_config("incartdb").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # headers + .atr files are the source
        assert spec.join_column == "record_name"

    def test_parses_all_three_comment_lines(self, tmp_path):
        from ecgbench.labels.incartdb import parse_header_comments

        (tmp_path / "I01.hea").write_text(self.HEADER_WITH_DX, encoding="utf-8")
        fields = parse_header_comments(tmp_path / "I01.hea")

        assert fields["age"] == "65"
        assert fields["sex"] == "F"
        assert fields["diagnosis"] == "Coronary artery disease, arterial hypertension"
        # Zero-padded so patient2 and patient20 sort and group predictably.
        assert fields["patient_id"] == "patient01"
        assert fields["record_features"] == "PVCs, noise"

    def test_absent_diagnoses_token_is_not_a_parse_failure(self, tmp_path):
        """34 of 75 records omit <diagnoses> entirely; age/sex/patient must survive."""
        from ecgbench.labels.incartdb import parse_header_comments

        (tmp_path / "I08.hea").write_text(self.HEADER_NO_DX, encoding="utf-8")
        fields = parse_header_comments(tmp_path / "I08.hea")

        assert fields["age"] == "51"
        assert fields["sex"] == "F"
        assert fields["diagnosis"] == ""  # empty, not None, so the column stays str
        assert fields["patient_id"] == "patient04"
        assert fields["record_features"] == "ventricular bigeminy, ventricular couplets"

    def test_trailing_comma_is_stripped_from_features(self, tmp_path):
        """Several real records end their findings line with ', ' (e.g. I17)."""
        from ecgbench.labels.incartdb import parse_header_comments

        header = self.HEADER_WITH_DX.replace("# PVCs, noise", "# bradycardia, PVCs,  ")
        (tmp_path / "I17.hea").write_text(header, encoding="utf-8")

        assert parse_header_comments(tmp_path / "I17.hea")["record_features"] == (
            "bradycardia, PVCs"
        )

    def test_beat_counts_separate_rhythm_markers_from_beats(self, tmp_path):
        """'+' is a rhythm change, not a beat, and must not inflate n_beats."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        from ecgbench.labels.incartdb import count_beats

        wfdb.wrann(
            "I01", "atr",
            sample=np.array([100, 200, 300, 400, 500]),
            symbol=["N", "N", "V", "+", "R"],
            write_dir=str(tmp_path),
        )
        counts = count_beats(tmp_path / "I01")

        assert counts["beat_N"] == 2
        assert counts["beat_V"] == 1
        assert counts["beat_R"] == 1
        assert counts["n_beats"] == 4          # not 5
        assert counts["n_rhythm_changes"] == 1

    def test_missing_annotation_file_does_not_kill_the_scan(self, tmp_path):
        """A record with no .atr yields zero counts rather than raising."""
        pytest.importorskip("wfdb")

        from ecgbench.labels.incartdb import count_beats

        counts = count_beats(tmp_path / "I99")

        assert counts["n_beats"] == 0
        assert counts["beat_N"] == 0

    def test_rare_classes_pool_on_patient_count_not_record_count(self):
        """Folds are grouped by patient, so a class needs enough PATIENTS."""
        import pandas as pd

        from ecgbench.labels.incartdb import OTHER, UNKNOWN, attach_stratify_class

        # 'Rare dx' has 12 records but only 2 patients — it cannot be spread over
        # 10 grouped folds, so it must pool even though 12 >= 10 records.
        df = pd.DataFrame({
            "diagnosis": ["Rare dx"] * 12 + ["Common dx"] * 10 + [""] * 10,
            "patient_id": (
                ["patient01"] * 6 + ["patient02"] * 6
                + [f"patient{i:02d}" for i in range(3, 13)]
                + [f"patient{i:02d}" for i in range(13, 23)]
            ),
        })
        out = attach_stratify_class(df)
        counts = out["stratify_class"].value_counts().to_dict()

        assert counts[OTHER] == 12          # Rare dx pooled: 2 patients
        assert counts["Common dx"] == 10    # 10 patients, kept
        assert counts[UNKNOWN] == 10        # empty diagnosis is its own class


class TestMimicIVECGLabels:
    """18 report columns and integer 'not measurable' sentinels."""

    MEASUREMENTS = {
        "study_id": [1, 2, 3],
        "cart_id": [10, 10, 11],
        "ecg_time": ["2180-07-23 08:44:00"] * 3,
        "report_0": ["Sinus rhythm.", "Atrial fibrillation", "  Sinus   Rhythm  "],
        # A populated line AFTER a blank one — the data dictionary warns about this.
        "report_1": [None, "Abnormal ECG", None],
        "report_2": ["Borderline ECG", None, "Otherwise normal ECG"],
        "rr_interval": [800, 65535, 900],
        "p_onset": [40, 29999, 42],
        "p_end": [150, 29999, 152],
        "qrs_onset": [200, 210, 205],
        "qrs_end": [290, 300, 299],
        "t_end": [610, 620, 615],
        "p_axis": [50, 32767, 55],
        "qrs_axis": [13, 20, -32768],
        "t_axis": [42, 40, 45],
        "bandwidth": ["0.5-40"] * 3,
        "filtering": ["60Hz"] * 3,
    }

    def _write(self, tmp_path, **overrides):
        data = dict(self.MEASUREMENTS)
        data.update(overrides)
        df = pd.DataFrame(data)
        # The real file has all 18 report columns; add the absent ones as empty.
        for i in range(18):
            if f"report_{i}" not in df.columns:
                df[f"report_{i}"] = None
        df.to_csv(tmp_path / "machine_measurements.csv", index=False)
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        from ecgbench.config import LabelConfig

        return replace(
            sample_config, slug="mimic_iv_ecg", record_id_column="study_id",
            labels=LabelConfig(source_csv="machine_measurements.csv",
                               join_column="study_id"),
        )

    def test_shipped_config_points_at_the_measurements_file(self):
        from ecgbench.config import load_config

        spec = load_config("mimic_iv_ecg").labels
        assert spec is not None and spec.available
        assert spec.source_csv == "machine_measurements.csv"
        assert spec.join_column == "study_id"

    def test_report_lines_join_across_a_blank_line(self, sample_config, tmp_path):
        """report_1 empty but report_2 populated must not truncate the report."""
        from ecgbench.labels.mimic_iv_ecg import load_labels

        df = load_labels(self._write(tmp_path), self._config(sample_config))

        assert df.loc[1, "report_text"] == "Sinus rhythm. | Borderline ECG"
        assert df.loc[2, "report_text"] == "Atrial fibrillation | Abnormal ECG"

    def test_primary_report_is_normalised(self, sample_config, tmp_path):
        """Trailing periods and irregular spacing split one class into several."""
        from ecgbench.labels.mimic_iv_ecg import load_labels, normalise_report_line

        df = load_labels(self._write(tmp_path), self._config(sample_config))

        # 'Sinus rhythm.' and '  Sinus   Rhythm  ' are the same statement.
        assert df.loc[1, "primary_report"] == "sinus rhythm"
        assert df.loc[3, "primary_report"] == "sinus rhythm"
        assert normalise_report_line("  ST changes.  ") == "st changes"
        assert normalise_report_line(None) == ""

    def test_integer_sentinels_become_nan(self, sample_config, tmp_path):
        """29999 / 32767 / -32768 / 65535 mean 'not measurable', not a value."""
        from ecgbench.labels.mimic_iv_ecg import load_labels

        df = load_labels(self._write(tmp_path), self._config(sample_config))

        assert pd.isna(df.loc[2, "rr_interval"])   # 65535
        assert pd.isna(df.loc[2, "p_onset"])       # 29999
        assert pd.isna(df.loc[2, "p_end"])         # 29999
        assert pd.isna(df.loc[2, "p_axis"])        # 32767
        assert pd.isna(df.loc[3, "qrs_axis"])      # -32768
        # Real values survive untouched.
        assert df.loc[1, "rr_interval"] == 800
        assert df.loc[1, "p_axis"] == 50

    def test_implausible_values_are_also_dropped(self, sample_config, tmp_path):
        """An axis of 4000 degrees is not a measurement even if it is not a sentinel."""
        from ecgbench.labels.mimic_iv_ecg import load_labels

        tree = self._write(tmp_path, qrs_axis=[13, 4000, -900])
        df = load_labels(tree, self._config(sample_config))

        assert df.loc[1, "qrs_axis"] == 13
        assert pd.isna(df.loc[2, "qrs_axis"])
        assert pd.isna(df.loc[3, "qrs_axis"])

    def test_qrs_duration_is_derived_and_propagates_nan(self, sample_config, tmp_path):
        from ecgbench.labels.mimic_iv_ecg import load_labels

        tree = self._write(tmp_path, qrs_onset=[200, 29999, 205])
        df = load_labels(tree, self._config(sample_config))

        assert df.loc[1, "qrs_duration"] == 90     # 290 - 200
        assert pd.isna(df.loc[2, "qrs_duration"])  # onset was a sentinel

    def test_raw_report_columns_are_preserved(self, sample_config, tmp_path):
        """report_text is a convenience; the raw lines must still be reachable."""
        from ecgbench.labels.mimic_iv_ecg import load_labels

        df = load_labels(self._write(tmp_path), self._config(sample_config))

        assert df.loc[1, "report_0"] == "Sinus rhythm."
        assert all(f"report_{i}" in df.columns for i in range(18))

    def test_rare_classes_pool_into_other(self, sample_config, tmp_path):
        import pandas as _pd

        from ecgbench.labels.mimic_iv_ecg import (
            MIN_CLASS_SIZE,
            OTHER,
            attach_stratify_class,
        )

        frame = _pd.DataFrame({
            "primary_report": ["sinus rhythm"] * MIN_CLASS_SIZE + ["rare finding"] * 3
        })
        out = attach_stratify_class(frame)
        counts = out["stratify_class"].value_counts().to_dict()

        assert counts["sinus rhythm"] == MIN_CLASS_SIZE
        assert counts[OTHER] == 3

    def test_missing_measurements_file_names_it_and_the_licence_reason(
        self, sample_config, tmp_path
    ):
        from ecgbench.labels.mimic_iv_ecg import load_labels

        with pytest.raises(LabelSourceMissingError) as excinfo:
            load_labels(tmp_path, self._config(sample_config))
        message = str(excinfo.value)
        assert "machine_measurements.csv" in message
        assert "credentialed" in message


class TestBrugadaHUCALabels:
    """Declarative labels off the shipped metadata.csv — no pipeline dependency."""

    def test_shipped_config_points_at_the_shipped_csv(self):
        """Deliberately metadata.csv, not the generated ecgbench_metadata.csv.

        Labels must work before `ecgbench splits` has ever run, unlike
        ecg_arrhythmia whose labels only exist afterwards.
        """
        from ecgbench.config import load_config

        spec = load_config("brugada_huca").labels
        assert spec is not None and spec.available
        assert spec.source_csv == "metadata.csv"
        assert spec.join_column == "patient_id"
        assert set(spec.columns) == {"brugada", "basal_pattern", "sudden_death"}

    def test_loads_the_three_clinical_columns(self, sample_config, tmp_path):
        from dataclasses import replace

        from ecgbench.config import LabelConfig

        pd.DataFrame({
            "patient_id": [188981, 251972, 265715],
            "basal_pattern": [1, 0, 0],
            "sudden_death": [0, 0, 1],
            "brugada": [1, 0, 2],
        }).to_csv(tmp_path / "metadata.csv", index=False)

        config = replace(
            sample_config, slug="brugada_huca", record_id_column="patient_id",
            labels=LabelConfig(source_csv="metadata.csv", join_column="patient_id",
                               columns=["brugada", "basal_pattern", "sudden_death"]),
        )
        df = load_labels(config, data_path=tmp_path)

        assert df.index.name == "patient_id"
        assert list(df.columns) == ["brugada", "basal_pattern", "sudden_death"]
        assert df.loc[188981, "brugada"] == 1
        assert df.loc[265715, "brugada"] == 2
        assert df.loc[265715, "sudden_death"] == 1

    def test_brugada_codes_have_documented_meanings(self):
        """The CSV ships bare 0/1/2; the meanings must live somewhere importable."""
        from ecgbench.splitting.strategies.brugada_huca import BRUGADA_CLASSES

        assert BRUGADA_CLASSES[0] == "healthy"
        assert BRUGADA_CLASSES[1] == "confirmed Brugada syndrome"
        assert BRUGADA_CLASSES[2] == "other/atypical"


class TestPTBXLPlusLabels:
    """PTB-XL+ is an annotation layer over PTB-XL's records, with an unkeyed table."""

    #: Deliberately NOT ascending, like the shipped file.
    IDS = [1, 21803, 21804, 7, 5]

    def _tree(self, tmp_path, ids=None, feature_rows=None, drop_key=False):
        ids = list(self.IDS if ids is None else ids)
        (tmp_path / "labels").mkdir(exist_ok=True)
        (tmp_path / "features").mkdir(exist_ok=True)

        pd.DataFrame({
            "ecg_id": ids,
            "statements": [str(["NSR", "NML"])] * len(ids),
            "statements_cat": [str(["NSR"])] * len(ids),
        }).to_csv(tmp_path / "labels/12sl_statements.csv", index=False)

        pd.DataFrame({
            "ecg_id": ids,
            "scp_codes": [str([("NORM", 100.0)])] * len(ids),
        }).to_csv(tmp_path / "labels/ptbxl_statements.csv", index=False)

        # The real 12sl feature table carries ecg_id at column 145 of 783 — present
        # but not first, which is what makes it easy to miss.
        n = len(ids) if feature_rows is None else feature_rows
        frame = pd.DataFrame({
            "HR__Global": [60 + i for i in range(n)],
            "ecg_id": ids[:n],
            "P_Area_I": [0.1 * i for i in range(n)],
        })
        if drop_key:
            frame = frame.drop(columns=["ecg_id"])
        frame.to_csv(tmp_path / "features/12sl_features.csv", index=False)

        pd.DataFrame({
            "ecg_id": ids,
            "RR_Mean_Global": [1000] * len(ids),
            # A name the 12sl table also uses — the real providers overlap heavily,
            # which is why prefixing is the default.
            "P_Area_I": [0.5] * len(ids),
        }).to_csv(tmp_path / "features/ecgdeli_features.csv", index=False)
        return tmp_path

    def test_12sl_key_is_found_by_name_not_position(self, tmp_path):
        """ecg_id sits mid-table in the real file (col 145 of 783), not first."""
        from ecgbench.labels.ptbxl_plus import load_features

        tree = self._tree(tmp_path)
        raw = pd.read_csv(tree / "features/12sl_features.csv", nrows=0).columns
        assert list(raw).index("ecg_id") != 0, "fixture must not put the key first"

        df = load_features(tree, "12sl")

        assert df.index.name == "ecg_id"
        # File order preserved, NOT sorted: the second half of the trap.
        assert list(df.index) == self.IDS
        assert sorted(df.index) != list(df.index)
        # Values stayed with their row, and the key is not left as a feature.
        assert df.loc[1, "HR__Global"] == 60
        assert df.loc[5, "HR__Global"] == 64
        assert "ecg_id" not in df.columns

    def test_fallback_keys_from_statements_if_the_column_ever_disappears(self, tmp_path):
        """Defensive path: v1.0.1 has the column, but a reissue might drop it."""
        from ecgbench.labels.ptbxl_plus import load_features

        df = load_features(self._tree(tmp_path, drop_key=True), "12sl")

        assert list(df.index) == self.IDS

    def test_row_count_mismatch_refuses_to_guess(self, tmp_path):
        """A positional join on mismatched lengths would corrupt every row."""
        from ecgbench.labels.ptbxl_plus import load_features

        # Key dropped so the row-order fallback engages, then lengths disagree.
        tree = self._tree(tmp_path, feature_rows=3, drop_key=True)

        with pytest.raises(ValueError, match="row-aligned"):
            load_features(tree, "12sl")

    def test_the_key_column_wins_over_the_statements_row_order(self, tmp_path):
        """If the two ever disagree, the table's own key is authoritative."""
        from ecgbench.labels.ptbxl_plus import load_features

        tree = self._tree(tmp_path)
        # A reissue where the feature rows are keyed differently from the
        # statements order: the key column must win, not the row position.
        pd.DataFrame({
            "HR__Global": [70, 71, 72, 73, 74],
            "ecg_id": [99, 98, 97, 96, 95],
        }).to_csv(tree / "features/12sl_features.csv", index=False)

        df = load_features(tree, "12sl")

        assert list(df.index) == [99, 98, 97, 96, 95]

    def test_keyed_feature_tables_use_their_own_column(self, tmp_path):
        from ecgbench.labels.ptbxl_plus import load_features

        df = load_features(self._tree(tmp_path), "ecgdeli")

        assert df.index.name == "ecg_id"
        assert list(df.index) == self.IDS

    def test_statement_literals_are_parsed(self, tmp_path):
        from ecgbench.labels.ptbxl_plus import load_statements

        st = load_statements(self._tree(tmp_path), "12sl")
        px = load_statements(self._tree(tmp_path), "ptbxl")

        assert st.loc[1, "statements"] == ["NSR", "NML"]
        assert px.loc[1, "scp_codes"] == [("NORM", 100.0)]

    def test_combined_frame_prefixes_providers(self, tmp_path):
        """The three feature sets share column names, so collisions must be impossible."""
        from ecgbench.labels.ptbxl_plus import load_ptbxl_plus

        df = load_ptbxl_plus(self._tree(tmp_path), features=("12sl", "ecgdeli"))

        assert "12sl_statements" in df.columns
        assert "ptbxl_scp_codes" in df.columns
        assert "12sl_HR__Global" in df.columns
        assert "ecgdeli_RR_Mean_Global" in df.columns
        assert not df.columns.duplicated().any()

    def test_unprefixed_collision_is_reported_not_silent(self, tmp_path):
        from ecgbench.labels.ptbxl_plus import load_ptbxl_plus

        # 12sl and ecgdeli both ship P_Area_I in this fixture, as the real
        # providers share many feature names.
        with pytest.raises(ValueError, match="Duplicate columns"):
            load_ptbxl_plus(self._tree(tmp_path), statements=(),
                            features=("12sl", "ecgdeli"), prefix=False)

    def test_median_beats_are_paths_only_with_provider_padding(self, tmp_path):
        """12sl pads to 5 digits, unig to 6 — and neither is decoded."""
        from ecgbench.labels.ptbxl_plus import median_beat_path

        for provider, stem in (("12sl", "00001_medians"), ("unig", "000001_medians")):
            d = tmp_path / "median_beats" / provider / "00000"
            d.mkdir(parents=True)
            (d / f"{stem}.hea").write_text("x", encoding="utf-8")
            assert median_beat_path(tmp_path, 1, provider).name == stem

        # Absent records return None rather than a path that does not exist.
        assert median_beat_path(tmp_path, 999999, "unig") is None

    def test_missing_source_names_the_release(self, tmp_path):
        from ecgbench.labels.ptbxl_plus import load_statements

        with pytest.raises(LabelSourceMissingError, match="ptb-xl-plus"):
            load_statements(tmp_path, "ptbxl")

    def test_there_is_deliberately_no_ptbxl_plus_config(self):
        """A config would let `ecgbench splits` build a second partition of PTB-XL."""
        from ecgbench.config import list_available_configs

        assert "ptbxl_plus" not in list_available_configs()
        assert "ptb_xl_plus" not in list_available_configs()


class TestMimicIVECGExtICDLabels:
    """Ext-ICD is an ICD-10 label layer over MIMIC-IV-ECG's records."""

    #: Deliberately not ascending, and one row with no linked diagnosis at all.
    STUDY_IDS = [40689238, 44458630, 49036311, 45090959, 48446569]

    def _table(self, tmp_path, gender=None):
        """A five-row stand-in for records_w_diag_icd10.csv."""
        from ecgbench.labels.mimic_iv_ecg_ext_icd import SOURCE_CSV

        pd.DataFrame({
            "study_id": self.STUDY_IDS,
            "subject_id": [10000032, 10000032, 10000032, 10000117, 10000117],
            "ecg_time": ["2180-07-23 08:44:00"] * 5,
            # 'W19XXXA' carries the trailing placeholder Xs; 'I2510' has real
            # superclasses; the fourth row has no linked diagnosis at all.
            "ed_diag_ed": [str(["R4182"]), str(["R4182"]), str([]), str([]),
                           str(["W19XXXA"])],
            "ed_diag_hosp": [str([])] * 5,
            "hosp_diag_hosp": [str([])] * 5,
            "all_diag_hosp": [str(["I2510"]), str(["I2510"]), str(["I2510"]),
                              str([]), str([])],
            "all_diag_all": [str(["I2510", "E785"]), str(["I2510"]),
                             str(["I2510", "W19XXXA"]), str([]), str(["E785"])],
            "gender": gender or ["F", "F", "F", "M", "missing"],
            "age": [52.0, 52.0, 52.0, 55.0, 57.0],
            "anchor_age": [52.0, 52.0, 52.0, 48.0, 48.0],
            "anchor_year": [2180.0] * 5,
            "dod": [None] * 5,
            "ecg_no_within_stay": [0, 1, 0, -1, 0],
            "ecg_taken_in_ed": [True, True, False, False, True],
            "ecg_taken_in_hosp": [False, False, True, False, False],
            "ecg_taken_in_ed_or_hosp": [True, True, True, False, True],
            "fold": [0, 0, 0, 18, 19],
            "strat_fold": [3, 3, 3, 18, 19],
        }).to_csv(tmp_path / SOURCE_CSV, index=False)
        return tmp_path

    def test_indexed_by_the_host_dataset_key_in_file_order(self, tmp_path):
        """The index must be MIMIC-IV-ECG's study_id so a reindex onto it works."""
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd

        df = load_ext_icd(self._table(tmp_path))

        assert df.index.name == "study_id"
        assert list(df.index) == self.STUDY_IDS
        assert "study_id" not in df.columns
        assert df.loc[40689238, "all_diag_all"] == ["I2510", "E785"]

    def test_empty_diagnoses_are_lists_not_nulls(self, tmp_path):
        """41.5% of the real table has no linked diagnosis; that is a value."""
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd

        df = load_ext_icd(self._table(tmp_path))

        assert df.loc[45090959, "all_diag_all"] == []
        assert df["all_diag_all"].notna().all()  # notna() cannot find them

    def test_missing_gender_marker_becomes_nan(self, tmp_path):
        """gender encodes missing as the string 'missing', not as a null."""
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd

        df = load_ext_icd(self._table(tmp_path))

        assert pd.isna(df.loc[48446569, "gender"])
        assert df["gender"].isna().sum() == 1
        assert "missing" not in set(df["gender"].dropna())

    def test_trailing_placeholder_xs_are_stripped_before_propagation(self):
        """The step that makes the published 1,076-code set reproducible."""
        from ecgbench.labels.mimic_iv_ecg_ext_icd import propagate_superclasses

        # W19XXXA -> W19XX (truncate) -> W19 (strip Xs). Without the strip this
        # would also yield 'W19X' and 'W19XX', and the label set comes out 1089.
        assert propagate_superclasses(["W19XXXA"]) == ["W19"]
        assert propagate_superclasses(["I2510"]) == ["I25", "I251", "I2510"]
        # Codes with no three-character category contribute nothing.
        assert propagate_superclasses(["A1"]) == []

    def test_label_set_counts_records_and_applies_the_threshold(self, tmp_path):
        from ecgbench.labels.mimic_iv_ecg_ext_icd import label_set, load_ext_icd

        df = load_ext_icd(self._table(tmp_path))

        # I25/I251/I2510 appear in 3 records, E78/E785 in 2, W19 in 1.
        assert label_set(df, min_count=3) == ["I25", "I251", "I2510"]
        assert label_set(df, min_count=1) == [
            "I25", "I251", "I2510", "E78", "E785", "W19",
        ]
        # Ordered by descending record count, so index 0 is the most common code.
        assert label_set(df, min_count=2)[0] in {"I25", "I251", "I2510"}

    def test_multi_hot_credits_superclasses(self, tmp_path):
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd, multi_hot

        df = load_ext_icd(self._table(tmp_path))
        targets = multi_hot(df, ["I25", "I251", "I2510", "E785", "W19"])

        assert list(targets.index) == self.STUDY_IDS
        # A record coded I2510 is positive for its two parent categories too.
        assert targets.loc[44458630].tolist() == [1, 1, 1, 0, 0]
        assert targets.loc[49036311].tolist() == [1, 1, 1, 0, 1]
        # No diagnosis -> all zero, not dropped.
        assert targets.loc[45090959].sum() == 0

    def test_ecg_subsets_are_the_benchmark_subsets(self, tmp_path):
        from ecgbench.labels.mimic_iv_ecg_ext_icd import ecg_subset, load_ext_icd

        df = load_ext_icd(self._table(tmp_path))

        assert len(ecg_subset(df, "ALL")) == 4
        assert len(ecg_subset(df, "ED")) == 3
        assert len(ecg_subset(df, "HOSP")) == 1
        # Case-insensitive, and an unknown subset is refused rather than empty.
        assert len(ecg_subset(df, "ed")) == 3
        with pytest.raises(ValueError, match="subset must be one of"):
            ecg_subset(df, "outpatient")

    def test_upstream_folds_are_zero_indexed_and_separate_from_ecgbenchs(self, tmp_path):
        """0-17 train, 18 val, 19 test -- not ECGBench's 1-indexed ten folds."""
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd, upstream_fold_split

        df = load_ext_icd(self._table(tmp_path))

        assert list(upstream_fold_split(df, "train").index) == self.STUDY_IDS[:3]
        assert list(upstream_fold_split(df, "val").index) == [45090959]
        assert list(upstream_fold_split(df, "test").index) == [48446569]
        # strat_fold is a different column and a different partition.
        assert list(upstream_fold_split(df, "train", stratified=True).index) == \
            self.STUDY_IDS[:3]
        with pytest.raises(ValueError, match="split must be one of"):
            upstream_fold_split(df, "validation")

    def test_prefix_keeps_ecg_time_from_colliding_with_the_host(self, tmp_path):
        """MIMIC-IV-ECG's own label frame also carries ecg_time."""
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd

        df = load_ext_icd(self._table(tmp_path), prefix="icd_")

        assert "icd_ecg_time" in df.columns
        assert "ecg_time" not in df.columns
        assert df.index.name == "study_id"  # the key itself is not prefixed

    def test_every_helper_accepts_the_prefix(self, tmp_path):
        """A prefixed frame must not silently break the helpers.

        The prefix cannot be inferred: 'fold' and 'strat_fold' share a suffix, so
        a suffix match on a prefixed frame is ambiguous. Hence prefix= everywhere.
        """
        from ecgbench.labels.mimic_iv_ecg_ext_icd import (
            ecg_subset,
            label_set,
            load_ext_icd,
            multi_hot,
            upstream_fold_split,
        )

        df = load_ext_icd(self._table(tmp_path), prefix="icd_")

        assert label_set(df, min_count=3, prefix="icd_") == ["I25", "I251", "I2510"]
        assert len(ecg_subset(df, "ED", prefix="icd_")) == 3
        assert list(upstream_fold_split(df, "test", prefix="icd_").index) == [48446569]
        targets = multi_hot(df, ["I25", "W19"], prefix="icd_")
        assert targets.loc[49036311].tolist() == [1, 1]

        # Without the prefix the column is genuinely absent, and the error says so
        # rather than returning an empty frame.
        with pytest.raises(ValueError, match="Pass prefix="):
            ecg_subset(df, "ED")

    def test_column_subset_keeps_the_key(self, tmp_path):
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd

        df = load_ext_icd(self._table(tmp_path), columns=["all_diag_all", "fold"])

        assert list(df.columns) == ["all_diag_all", "fold"]
        assert list(df.index) == self.STUDY_IDS

    def test_missing_source_names_the_release_and_the_agreement(self, tmp_path):
        from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd

        with pytest.raises(LabelSourceMissingError, match="mimic-iv-ecg-ext-icd-labels"):
            load_ext_icd(tmp_path)

    def test_there_is_deliberately_no_ext_icd_config(self):
        """A config would let `ecgbench splits` build a second MIMIC-IV-ECG partition."""
        from ecgbench.config import list_available_configs
        from ecgbench.labels import _custom_loaders

        available = list_available_configs()
        assert "mimic_iv_ecg_ext_icd" not in available
        assert "mimic_iv_ecg_ext_icd_labels" not in available
        # _custom_loaders maps *config* slugs to loaders, and there is no config.
        assert "mimic_iv_ecg_ext_icd" not in _custom_loaders()


class TestLeipzigHeartCenterLabels:
    """Leipzig ships two subject CSVs, six channel layouts and a malformed age."""

    def _tree(self, tmp_path):
        """A two-record stand-in: one child (19 channels), one adult (14)."""
        wfdb = pytest.importorskip("wfdb")

        pd.DataFrame({
            "subject_id": ["001", "007"],
            "file_name": ["x001", "x007"],
            "gender": ["M", "M"],
            "age": ["6.6", ".14.3"],          # the second is the shipped malformation
            "diagnosis": ["AVRT-WPW", "AVRT-PJRT"],
            "ap_loacation": ["right posteroseptal", ""],   # source misspelling
            "ecg_duration": ["0:00:02.0", "0:00:02.0"],
        }).to_csv(tmp_path / "children-subject-info.csv", index=False)

        pd.DataFrame({
            "subject_id": ["100"],
            "file_name": ["x100"],
            "gender": ["F"],
            "age": ["64.16"],
            "diagnosis": ["TOF with VT"],     # no ap_loacation column at all
            "ecg_duration": ["0:00:02.0"],
        }).to_csv(tmp_path / "adults-subject-info.csv", index=False)

        # Three layouts, mirroring the real ones: the child has ABL12 at index 12,
        # the adult puts RVA12 last (like x100), and x007 has no ABL12 at all.
        layouts = {
            "x001": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4",
                     "V5", "V6", "ABL12", "RVA12"],
            "x007": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4",
                     "V5", "V6", "RVA12"],
            "x100": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4",
                     "V5", "V6", "ABL12", "CS12", "RVA12"],
        }
        for record, names in layouts.items():
            n = len(names)
            signal = np.tile(np.arange(1954, dtype=np.float64)[:, None] / 1000.0, (1, n))
            wfdb.wrsamp(record, fs=977, units=["mV"] * n, sig_name=names,
                        p_signal=signal, fmt=["16"] * n, adc_gain=[2000.0] * n,
                        baseline=[0] * n, write_dir=str(tmp_path))
            # One of every kind of annotation this module counts. custom_labels
            # mirrors what the real .atr files declare — X and b are non-standard
            # WFDB symbols at label stores 42 and 43 — without which wfdb rewrites
            # them as '"' NOTE annotations and the aux string absorbs the symbol.
            wfdb.wrann(
                record, "atr",
                np.array([10, 20, 30, 40, 50, 60, 70]),
                np.array(["N", "X", "X", "/", "Q", "~", "+"]),
                aux_note=["N-Prex", "AVRT", "AFIB", "/V", "", "", "(N"],
                fs=977, write_dir=str(tmp_path),
                custom_labels=[(42, "X", "Tachycardias"), (43, "b", "AV-Block")],
            )
        return tmp_path

    def test_malformed_age_is_repaired_and_kept_verbatim(self):
        """x007 ships '.14.3', which no float parser accepts."""
        from ecgbench.labels.leipzig_heart_center_ecg import parse_age

        assert parse_age("6.6") == 6.6
        assert parse_age(".14.3") == 14.3      # single leading '.' stripped
        assert parse_age("") is None
        assert parse_age("not-a-number") is None
        # Only the one-stray-dot shape is repaired; anything else is not guessed.
        assert parse_age("..14.3") is None

    def test_duration_parses_the_h_m_s_form(self):
        from ecgbench.labels.leipzig_heart_center_ecg import parse_duration

        assert parse_duration("2:25:29.201") == pytest.approx(8729.201)
        assert parse_duration("0:1:17.659") == pytest.approx(77.659)
        assert parse_duration("garbage") is None

    def test_diagnosis_family_is_the_leading_token(self):
        from ecgbench.labels.leipzig_heart_center_ecg import diagnosis_family

        assert diagnosis_family("AVRT-WPW") == "AVRT"
        assert diagnosis_family("AVRT-PJRT") == "AVRT"
        assert diagnosis_family("AVNRT") == "AVNRT"
        assert diagnosis_family("TOF with VT") == "TOF"
        assert diagnosis_family("TOF without VT") == "TOF"
        assert diagnosis_family("") == "UNKNOWN"

    def test_channel_index_finds_channels_by_name_not_position(self):
        """Index 12 is ABL12, RVA12 or ART depending on the record."""
        from ecgbench.labels.leipzig_heart_center_ecg import channel_index

        child = "I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6|ABL12|RVA12|CS12"
        x100 = "I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6|ABL12|CS12|RVA12"
        x0028 = "I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6|ART|ABL12|RVA12"

        # The same channel sits at three different indices across the three layouts.
        assert channel_index(child, "RVA12") == 13
        assert channel_index(x100, "RVA12") == 14
        assert channel_index(x0028, "RVA12") == 14
        assert channel_index(x0028, "ABL12") == 13
        assert channel_index(child, "ABL12") == 12
        # Absent channels are None, not a wrong index.
        assert channel_index("I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6|RVA12", "ABL12") is None
        # A list works as well as the '|'-joined column.
        assert channel_index(["I", "II", "ABL12"], "ABL12") == 2

    def test_the_two_subject_csvs_are_joined_with_a_cohort_column(self, tmp_path):
        from ecgbench.labels.leipzig_heart_center_ecg import scan_records

        df = scan_records(self._tree(tmp_path))

        assert len(df) == 3
        # Sorted numerically, not lexically: x001, x007, x100.
        assert list(df["record_name"]) == ["x001", "x007", "x100"]
        assert df.set_index("record_name")["cohort"].to_dict() == {
            "x001": "child", "x007": "child", "x100": "adult",
        }
        # The children's misspelled column is exposed corrected, and the adults'
        # row — whose CSV has no such column at all — comes back null.
        assert "ap_loacation" not in df.columns
        assert df.set_index("record_name").loc["x001", "ap_location"] == "right posteroseptal"
        assert pd.isna(df.set_index("record_name").loc["x100", "ap_location"])

    def test_channel_layout_comes_from_each_records_own_header(self, tmp_path):
        from ecgbench.labels.leipzig_heart_center_ecg import channel_index, scan_records

        df = scan_records(self._tree(tmp_path)).set_index("record_name")

        assert df["n_signals"].to_dict() == {"x001": 14, "x007": 13, "x100": 15}
        assert df["sampling_rate"].unique().tolist() == [977]
        # n_iegm_channels is everything past the 12 ECG leads.
        assert df["n_iegm_channels"].to_dict() == {"x001": 2, "x007": 1, "x100": 3}
        # RVA12 is at a different index in each, which is the whole point.
        assert channel_index(df.loc["x001", "channel_names"], "RVA12") == 13
        assert channel_index(df.loc["x100", "channel_names"], "RVA12") == 14
        assert channel_index(df.loc["x007", "channel_names"], "ABL12") is None

    def test_beat_total_excludes_unclassifiable_and_non_beat_marks(self, tmp_path):
        """The release's 113,924 counts only the classes its README tabulates."""
        from ecgbench.labels.leipzig_heart_center_ecg import scan_records

        df = scan_records(self._tree(tmp_path)).set_index("record_name")
        row = df.loc["x001"]

        # 7 annotations per record: N, X, X, /, Q, ~, + -> 4 tabulated beats.
        assert row["n_annotations"] == 7
        assert row["n_beats"] == 4
        assert row["n_unclassifiable"] == 1     # Q
        assert row["n_quality_marks"] == 1      # ~ — not a beat at all
        assert row["n_rhythm_changes"] == 1     # +
        assert row["n_beats"] + row["n_unclassifiable"] + row["n_quality_marks"] \
            + row["n_rhythm_changes"] == row["n_annotations"]

    def test_aux_strings_are_counted_per_category(self, tmp_path):
        from ecgbench.labels.leipzig_heart_center_ecg import scan_records

        row = scan_records(self._tree(tmp_path)).set_index("record_name").loc["x001"]

        # The two X beats name their mechanism; the totals must match beat_X.
        assert row["tachy_AVRT"] == 1
        assert row["tachy_AFIB"] == 1
        assert row["beat_X"] == 2
        assert row["tachy_AVRT"] + row["tachy_AFIB"] == row["beat_X"]
        assert row["aux_preexcited_N"] == 1      # N-Prex qualifies an N beat
        assert row["aux_paced_ventricular"] == 1  # /V qualifies a / beat
        assert row["rhythm_sinus"] == 1           # (N on the + marker

    def test_stratify_class_pools_families_below_the_floor(self, tmp_path):
        from ecgbench.labels.leipzig_heart_center_ecg import (
            attach_stratify_class,
            scan_records,
        )

        df = attach_stratify_class(scan_records(self._tree(tmp_path)))

        # In this 3-record fixture every family is below MIN_CLASS_RECORDS, so all
        # of them pool — the mechanism the real 39-record dataset never triggers.
        assert set(df["stratify_class"]) == {"OTHER"}
        # The un-pooled family survives alongside it, so nothing is lost.
        assert df.set_index("record_name")["diagnosis_family"].to_dict() == {
            "x001": "AVRT", "x007": "AVRT", "x100": "TOF",
        }

    def test_load_labels_is_indexed_by_the_configs_record_id(self, tmp_path, sample_config):
        from dataclasses import replace

        from ecgbench.labels.leipzig_heart_center_ecg import load_labels

        config = replace(sample_config, record_id_column="record_name")
        df = load_labels(self._tree(tmp_path), config)

        assert df.index.name == "record_name"
        assert list(df.index) == ["x001", "x007", "x100"]
        assert "stratify_class" in df.columns

    def test_missing_subject_csvs_name_the_release(self, tmp_path):
        from ecgbench.labels.leipzig_heart_center_ecg import scan_records

        with pytest.raises(LabelSourceMissingError, match="leipzig-heart-center-ecg"):
            scan_records(tmp_path)


class TestNorwegianAthleteECGLabels:
    """Two free-text interpretations per header, with commas inside statements."""

    # Real ath_001 content. Note the double space before "Normal ECG" — the file
    # has it, and it must not become an empty statement.
    HEADER = (
        "ath_001 12 500 5000\n"
        "ath_001.dat 16 50000/mV 16 0 10251 49595 0 I\n"
        "#SL12: Sinus bradycardia with marked sinus arrhythmia, Right axis"
        " deviation, Borderline ECG\n"
        "#C: Sinus arrhythmia,  Normal ECG\n"
    )

    def _tree(self, tmp_path, records):
        """Write a RECORDS file plus one .hea per entry of {name: header text}."""
        (tmp_path / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
        for name, text in records.items():
            (tmp_path / f"{name}.hea").write_text(text, encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config,
            slug="norwegian_athlete_ecg",
            record_id_column="record_name",
            url="https://physionet.org/content/norwegian-athlete-ecg/1.0.0/",
        )

    def test_labels_come_from_headers_not_a_csv(self):
        from ecgbench.config import load_config

        spec = load_config("norwegian_athlete_ecg").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # the .hea comment lines are the source
        assert spec.join_column == "record_name"

    def test_parses_both_interpretations(self, tmp_path, sample_config):
        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        path = self._tree(tmp_path, {"ath_001": self.HEADER})
        df = load_labels(path, self._config(sample_config))

        assert df.index.name == "record_name"
        assert list(df.index) == ["ath_001"]
        row = df.loc["ath_001"]
        assert row["sl12_findings"] == [
            "Sinus bradycardia with marked sinus arrhythmia", "Right axis deviation",
        ]
        assert row["sl12_verdict"] == "Borderline ECG"
        # The double space must not leave a stray empty finding behind.
        assert row["cardiologist_findings"] == ["Sinus arrhythmia"]
        assert row["cardiologist_verdict"] == "Normal ECG"
        assert row["cardiologist_primary_rhythm"] == "Sinus arrhythmia"
        # Raw strings are kept verbatim for both sources.
        assert row["cardiologist_raw"] == "Sinus arrhythmia,  Normal ECG"

    def test_statements_containing_commas_are_not_shattered(self):
        """The trap: 3 GE statements have commas of their own (ath_024, ath_027)."""
        from ecgbench.labels.norwegian_athlete_ecg import split_statements

        line = (
            "Sinus bradycardia, Nonspecific intraventricular conduction delay, "
            "ST elevation, consider early repolarization, pericarditis, or injury, "
            "Abnormal ECG"
        )
        assert split_statements(line) == [
            "Sinus bradycardia",
            "Nonspecific intraventricular conduction delay",
            "ST elevation, consider early repolarization, pericarditis, or injury",
            "Abnormal ECG",
        ]
        # A naive split would report 7 statements instead of 4.
        assert len(line.split(",")) == 7

    def test_lowercase_fragments_can_be_real_statements(self):
        """Why capitalisation cannot be used to detect continuations.

        ath_005 and ath_017 write genuine findings in lowercase, so the
        'lowercase means continuation' heuristic would silently merge them.
        """
        from ecgbench.labels.norwegian_athlete_ecg import split_statements

        assert split_statements(
            "Sinus bradycardia, normal sinus rhythm, First degree AV block, Normal ECG"
        ) == [
            "Sinus bradycardia", "normal sinus rhythm", "First degree AV block",
            "Normal ECG",
        ]

    def test_critical_and_acute_alerts_leave_the_findings_list(self, tmp_path, sample_config):
        """ath_007-style header: asterisk-wrapped alerts become their own columns."""
        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        header = (
            "ath_007 12 500 5000\n"
            "ath_007.dat 16 50000/mV 16 0 646 573 0 I\n"
            "#SL12: ***Critical test result: STEMI, Normal sinus rhythm, Pulmonary"
            " disease pattern, ** ** ACUTE MI/STEMI** **, Abnormal ECG\n"
            "#C: Normal sinus rhythm, Normal ECG\n"
        )
        df = load_labels(
            self._tree(tmp_path, {"ath_007": header}), self._config(sample_config)
        )
        row = df.loc["ath_007"]

        assert row["sl12_critical_test_result"] == "STEMI"
        assert row["sl12_acute_alert"] == "ACUTE MI/STEMI"  # asterisks stripped
        assert row["sl12_findings"] == [
            "Normal sinus rhythm", "Pulmonary disease pattern",
        ]
        assert row["sl12_verdict"] == "Abnormal ECG"
        # The dataset's headline quantity: SL12 flags what the cardiologist clears.
        assert row["cardiologist_verdict"] == "Normal ECG"
        assert bool(row["sl12_overcalls"]) is True
        assert bool(row["verdicts_match"]) is False

    def test_ekg_misspelling_normalises_but_the_raw_form_survives(self, tmp_path, sample_config):
        """ath_010 ends 'Abnormal EKG'. Dropping it would lose a whole record."""
        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        header = self.HEADER.replace("Borderline ECG", "Abnormal EKG")
        df = load_labels(
            self._tree(tmp_path, {"ath_001": header}), self._config(sample_config)
        )

        assert df.loc["ath_001", "sl12_verdict"] == "Abnormal ECG"
        assert df.loc["ath_001", "sl12_verdict_raw"] == "Abnormal EKG"

    def test_unknown_verdict_raises_rather_than_going_silently_missing(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        header = self.HEADER.replace("Borderline ECG", "Inconclusive tracing")
        with pytest.raises(ValueError, match="unrecognised .* verdict"):
            load_labels(
                self._tree(tmp_path, {"ath_001": header}), self._config(sample_config)
            )

    def test_unknown_opening_rhythm_raises_because_it_is_the_split_label(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        header = self.HEADER.replace("#C: Sinus arrhythmia, ", "#C: Atrial fibrillation, ")
        with pytest.raises(ValueError, match="not a known rhythm"):
            load_labels(
                self._tree(tmp_path, {"ath_001": header}), self._config(sample_config)
            )

    def test_missing_interpretation_line_raises(self, tmp_path, sample_config):
        """Every record carries both lines; one missing means a truncated copy."""
        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        header = "\n".join(
            line for line in self.HEADER.splitlines() if not line.startswith("#C:")
        ) + "\n"
        with pytest.raises(ValueError, match=r"missing the \['C'\]"):
            load_labels(
                self._tree(tmp_path, {"ath_001": header}), self._config(sample_config)
            )

    def test_header_listed_in_records_but_absent_names_the_release(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        (tmp_path / "RECORDS").write_text("ath_001\nath_002\n", encoding="utf-8")
        (tmp_path / "ath_001.hea").write_text(self.HEADER, encoding="utf-8")

        with pytest.raises(LabelSourceMissingError, match="norwegian-athlete-ecg"):
            load_labels(tmp_path, self._config(sample_config))


class TestMHDEffectECGMRILabels:
    """No diagnosis to predict: the labels are acquisition conditions + QRS counts."""

    #: A 3-channel header keeps the fixtures small. Note "Positon" — the source's
    #: own misspelling, which the parser must match.
    def _header(self, record, *, field="3T", scanner="Siemens Magnetom Skyra",
                b0="Horizontal", position="Feet first (Ff)", sex="Male",
                age="27years", weight="75kg", height="190cm", n_samples=25000,
                recorder="Getemed CM 3000, 12-lead Holter ECG",
                lead_config="Diagnostic 12 lead ECG"):
        return (
            f"{record} 3 1024 {n_samples}\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 I\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 II\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 III\n"
            "# \n"
            "#*Technical parameters of the MR scanner:\n"
            f"#--Magnetic field strength:{field}\n"
            f"#--MR scanner:{scanner}\n"
            f"#--Orientation of the static magnetic field (B0):{b0}\n"
            "#*Technical parameters of the ECG hardware: \n"
            f"#--ECG recorder:{recorder}\n"
            "#--ADC resolution:12bit\n"
            "#--ADC input voltage range:+/-6mV\n"
            f"#--ECG lead configuration:{lead_config}\n"
            "#*Information about the subject: \n"
            f"#--Sex:{sex}\n"
            f"#--Age:{age}\n"
            f"#--Weight:{weight}\n"
            f"#--Height:{height}\n"
            f"#--Positon in the scanner:{position}\n"
            "#--Respiration:Spontaneous respiration\n"
        )

    def _tree(self, tmp_path, headers):
        """Write RECORDS plus one .hea per {record: header} entry."""
        (tmp_path / "RECORDS").write_text("\n".join(headers) + "\n", encoding="utf-8")
        for record, text in headers.items():
            (tmp_path / f"{record}.hea").write_text(text, encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config,
            slug="mhd_effect_ecg_mri",
            record_id_column="record_name",
            patient_id_column="subject_key",
            default_sampling_rate=1024,
            signal_path_columns={1024: "signal_path"},
            url="https://physionet.org/content/mhd-effect-ecg-mri/1.0.0/",
        )

    def test_labels_come_from_headers_not_a_csv(self):
        from ecgbench.config import load_config

        spec = load_config("mhd_effect_ecg_mri").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # .hea comment block + .qrs files
        assert spec.join_column == "record_name"

    def test_parse_record_name_splits_field_subject_and_position(self):
        from ecgbench.labels.mhd_effect_ecg_mri import parse_record_name

        parsed = parse_record_name("ECGMRI3T04Ff")
        assert parsed["scanner_field_T"] == 3
        assert parsed["field_strength_T"] == 3
        assert parsed["subject_number"] == "04"        # zero-padded, stays a string
        assert parsed["scanner_subject_slot"] == "3T04"
        assert parsed["position"] == "Feet first"
        assert parsed["run"] == 1
        assert parsed["is_reference"] is False
        assert parsed["condition"] == "3T"

    def test_reference_records_get_zero_field_but_keep_their_scanner(self):
        """B0 is 0 outside the bore, yet the session still belongs to a scanner."""
        from ecgbench.labels.mhd_effect_ecg_mri import parse_record_name

        parsed = parse_record_name("ECGMRI7T02Out")
        assert parsed["is_reference"] is True
        assert parsed["field_strength_T"] == 0     # what the subject was exposed to
        assert parsed["scanner_field_T"] == 7      # which session it belongs to
        assert parsed["condition"] == "reference"

    def test_repeated_runs_are_numbered(self):
        """ECGMRI3T09Ff1/2/3 are three feet-first runs of one subject."""
        from ecgbench.labels.mhd_effect_ecg_mri import parse_record_name

        assert [parse_record_name(f"ECGMRI3T09Ff{n}")["run"] for n in (1, 2, 3)] == [1, 2, 3]
        assert parse_record_name("ECGMRI3T09Hf")["run"] == 1

    def test_unparseable_record_name_raises(self):
        from ecgbench.labels.mhd_effect_ecg_mri import parse_record_name

        with pytest.raises(ValueError, match="does not match the documented convention"):
            parse_record_name("someRecord01")
        with pytest.raises(ValueError, match="unknown position suffix"):
            parse_record_name("ECGMRI3T04Zz")

    def test_parses_the_misspelled_position_key(self, tmp_path):
        """The header key is '#--Positon in the scanner', not 'Position'."""
        from ecgbench.labels.mhd_effect_ecg_mri import parse_header_comments

        record = "ECGMRI3T04Ff"
        (tmp_path / f"{record}.hea").write_text(self._header(record), encoding="utf-8")
        fields = parse_header_comments(tmp_path / f"{record}.hea")

        assert fields["position_header"] == "Feet first (Ff)"
        assert fields["mr_scanner"] == "Siemens Magnetom Skyra"
        assert fields["lead_config"] == "Diagnostic 12 lead ECG"
        assert fields["respiration"] == "Spontaneous respiration"
        assert fields["n_signals"] == 3
        assert fields["sampling_rate"] == 1024
        assert fields["channel_names"] == "I|II|III"
        assert fields["duration_seconds"] == round(25000 / 1024, 3)

    def test_subject_key_reunites_the_same_person_across_scanners(self, tmp_path, sample_config):
        """The real trap: subject numbers are per-scanner, so 3T01 != 1T01.

        Grouping on the filename number would split one person across folds, which
        is leakage in a dataset built to compare field strengths.
        """
        from ecgbench.labels.mhd_effect_ecg_mri import load_labels

        same = dict(sex="Male", age="27years", weight="75kg", height="190cm")
        other = dict(sex="Female", age="29years", weight="60kg", height="165cm")
        tree = self._tree(tmp_path, {
            "ECGMRI1T01Sup": self._header("ECGMRI1T01Sup", field="1T", position="Supine", **same),
            "ECGMRI3T02Ff": self._header("ECGMRI3T02Ff", **same),
            "ECGMRI7T05Ff": self._header("ECGMRI7T05Ff", field="7T", **same),
            "ECGMRI3T01Ff": self._header("ECGMRI3T01Ff", **other),
        })
        df = load_labels(tree, self._config(sample_config))

        # Three filename slots, one person.
        assert df.loc["ECGMRI1T01Sup", "subject_key"] == df.loc["ECGMRI3T02Ff", "subject_key"]
        assert df.loc["ECGMRI7T05Ff", "subject_key"] == df.loc["ECGMRI3T02Ff", "subject_key"]
        # Same subject NUMBER, different person — must not collide.
        assert df.loc["ECGMRI3T01Ff", "subject_key"] != df.loc["ECGMRI1T01Sup", "subject_key"]
        assert df["scanner_subject_slot"].nunique() == 4
        assert df["subject_key"].nunique() == 2

    def test_position_disagreement_is_flagged_not_resolved(self, tmp_path, sample_config):
        """ECGMRI3T01Hf's filename says head-first; its header says feet-first."""
        from ecgbench.labels.mhd_effect_ecg_mri import load_labels

        tree = self._tree(tmp_path, {
            "ECGMRI3T01Hf": self._header("ECGMRI3T01Hf", position="Feet first (Ff)"),
            "ECGMRI3T01Ff": self._header("ECGMRI3T01Ff", position="Feet first (Ff)"),
        })
        df = load_labels(tree, self._config(sample_config))

        assert bool(df.loc["ECGMRI3T01Hf", "position_disagrees"]) is True
        assert df.loc["ECGMRI3T01Hf", "position"] == "Head first"          # filename
        assert df.loc["ECGMRI3T01Hf", "position_header"] == "Feet first (Ff)"
        assert bool(df.loc["ECGMRI3T01Ff", "position_disagrees"]) is False

    def test_reference_record_naming_a_field_strength_is_flagged(self, tmp_path, sample_config):
        """ECGMRI1T01Out says field '1T' though it was recorded outside the bore.

        Filtering references on field_strength_header would silently miss it, so
        is_reference/condition come from the filename instead.
        """
        from ecgbench.labels.mhd_effect_ecg_mri import load_labels

        tree = self._tree(tmp_path, {
            # The inconsistent one: position says outside, field strength says 1T.
            "ECGMRI1T01Out": self._header(
                "ECGMRI1T01Out", field="1T", b0="Vertical",
                position="Outside the scanner",
            ),
            # The other nine look like this.
            "ECGMRI7T02Out": self._header(
                "ECGMRI7T02Out", field="Outside the scanner",
                b0="Outside the scanner", position="Outside the scanner",
            ),
        })
        df = load_labels(tree, self._config(sample_config))

        assert bool(df.loc["ECGMRI1T01Out", "reference_header_agrees"]) is False
        assert bool(df.loc["ECGMRI7T02Out", "reference_header_agrees"]) is True
        # Both are still recognised as references, which is the point.
        assert df["is_reference"].all()
        assert set(df["condition"]) == {"reference"}

    def test_demographics_are_parsed_to_numbers_and_kept_raw(self, tmp_path, sample_config):
        from ecgbench.labels.mhd_effect_ecg_mri import load_labels

        tree = self._tree(tmp_path, {"ECGMRI3T04Ff": self._header("ECGMRI3T04Ff")})
        row = load_labels(tree, self._config(sample_config)).loc["ECGMRI3T04Ff"]

        assert row["age_raw"] == "27years" and row["age"] == 27.0
        assert row["weight_raw"] == "75kg" and row["weight"] == 75.0
        assert row["height_raw"] == "190cm" and row["height"] == 190.0

    def test_qrs_counts_separate_unexpected_symbols(self, tmp_path):
        """All 14,950 marks in v1.0.0 are 'N', and they are POSITIONS not classes."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        from ecgbench.labels.mhd_effect_ecg_mri import count_qrs

        wfdb.wrann("ECGMRI3T04Ff", "qrs",
                   sample=np.array([100, 200, 300, 400]),
                   symbol=["N", "N", "N", "V"], write_dir=str(tmp_path))
        counts = count_qrs(tmp_path / "ECGMRI3T04Ff")

        assert counts["n_qrs"] == 3
        assert counts["n_qrs_other"] == 1

    def test_missing_annotation_file_does_not_kill_the_scan(self, tmp_path):
        pytest.importorskip("wfdb")

        from ecgbench.labels.mhd_effect_ecg_mri import count_qrs

        assert count_qrs(tmp_path / "ECGMRI9T99Ff") == {"n_qrs": 0, "n_qrs_other": 0}

    def test_header_listed_in_records_but_absent_names_the_release(self, tmp_path, sample_config):
        from ecgbench.labels.mhd_effect_ecg_mri import scan_records

        (tmp_path / "RECORDS").write_text("ECGMRI3T04Ff\nECGMRI3T05Ff\n", encoding="utf-8")
        (tmp_path / "ECGMRI3T04Ff.hea").write_text(
            self._header("ECGMRI3T04Ff"), encoding="utf-8"
        )
        with pytest.raises(LabelSourceMissingError, match="mhd-effect-ecg-mri"):
            scan_records(tmp_path, self._config(sample_config))


class TestWCTECGLabels:
    """Demographics and one free-text admission diagnosis, from the headers."""

    # Real patient001/seg01 content, truncated to two of the 37 signal lines. The
    # diagnosis carries byte 0xA0 where a dash belongs — that is what the release
    # ships, and it is why the loader decodes cp1252 rather than utf-8.
    DIAGNOSIS = "Non ST\xa0segment\xa0elevation myocardial infarction (NSTEMI)"
    HEADER = (
        "seg01 37 800 8001\n"
        "seg01.dat 16 36213.4604(-6137)/mV 0 0 500 -11346 0 I-Raw\n"
        "seg01.dat 16 145039.7107(2528)/mV 0 0 -3436 -23891 0 WCT\n"
        "\n"
        "#Age: 46\n"
        "#Sex: M\n"
        f"#Diagnosis report: {DIAGNOSIS}\n"
    )

    def _tree(self, tmp_path, records):
        """Write RECORDS plus one patientNNN/segMM.hea per {path: header text}."""
        (tmp_path / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
        for name, text in records.items():
            path = tmp_path / f"{name}.hea"
            path.parent.mkdir(parents=True, exist_ok=True)
            # cp1252, like the release — a utf-8 write would not reproduce byte 0xA0.
            path.write_text(text, encoding="cp1252")
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config,
            slug="wctecgdb",
            record_id_column="record_name",
            url="https://physionet.org/content/wctecgdb/1.0.1/",
        )

    def test_labels_come_from_headers_not_a_csv(self):
        from ecgbench.config import load_config

        spec = load_config("wctecgdb").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # the .hea comment lines are the source
        assert spec.join_column == "record_name"

    def test_record_id_flattens_the_patient_directory(self, tmp_path, sample_config):
        """seg01 exists in all 92 patient directories, so it cannot be the id."""
        from ecgbench.labels.wctecgdb import load_labels

        path = self._tree(
            tmp_path,
            {"patient001/seg01": self.HEADER, "patient002/seg01": self.HEADER},
        )
        df = load_labels(path, self._config(sample_config))

        assert df.index.name == "record_name"
        assert list(df.index) == ["patient001_seg01", "patient002_seg01"]
        assert df["patient_id"].tolist() == ["patient001", "patient002"]
        assert df["segment"].tolist() == ["seg01", "seg01"]

    def test_demographics_and_diagnosis_are_parsed(self, tmp_path, sample_config):
        from ecgbench.labels.wctecgdb import load_labels

        df = load_labels(
            self._tree(tmp_path, {"patient001/seg01": self.HEADER}),
            self._config(sample_config),
        )
        row = df.loc["patient001_seg01"]

        assert row["age"] == 46
        assert row["sex"] == "M"
        # The non-breaking spaces are folded, and the raw string survives verbatim.
        assert row["diagnosis"] == (
            "Non ST segment elevation myocardial infarction (NSTEMI)"
        )
        assert row["diagnosis_raw"] == self.DIAGNOSIS
        assert "\xa0" in row["diagnosis_raw"]
        assert bool(row["diagnosis_reported"]) is True
        assert row["diagnosis_group"] == "Myocardial infarction"
        assert row["reconstructed_precordials"] == []
        assert bool(row["has_reconstructed_precordials"]) is False

    def test_windows_1252_headers_do_not_need_replacement_characters(
        self, tmp_path, sample_config
    ):
        """A utf-8 read of these headers either raises or produces U+FFFD."""
        from ecgbench.labels.wctecgdb import load_labels

        path = self._tree(tmp_path, {"patient001/seg01": self.HEADER})
        raw = (path / "patient001/seg01.hea").read_bytes()
        assert b"\xa0" in raw
        with pytest.raises(UnicodeDecodeError):
            raw.decode("utf-8")

        df = load_labels(path, self._config(sample_config))
        assert "�" not in df.loc["patient001_seg01", "diagnosis"]

    def test_misspellings_are_corrected_and_the_raw_form_survives(
        self, tmp_path, sample_config
    ):
        """Four real variants; leaving them apart would split one class into two."""
        from ecgbench.labels.wctecgdb import load_labels, normalise_diagnosis

        assert normalise_diagnosis("Atypica chest pain") == "Atypical chest pain"
        assert normalise_diagnosis("sinus bradycardia") == "Sinus bradycardia"
        assert normalise_diagnosis("Type 2 Myocaridal infarctoin") == (
            "Type 2 myocardial infarction"
        )
        assert normalise_diagnosis("Congestive Cardic failure (CCF)") == (
            "Congestive cardiac failure (CCF)"
        )

        header = self.HEADER.replace(self.DIAGNOSIS, "Atypica chest pain")
        df = load_labels(
            self._tree(tmp_path, {"patient001/seg01": header}),
            self._config(sample_config),
        )
        assert df.loc["patient001_seg01", "diagnosis"] == "Atypical chest pain"
        assert df.loc["patient001_seg01", "diagnosis_raw"] == "Atypica chest pain"

    def test_not_reported_is_a_value_not_a_blank(self, tmp_path, sample_config):
        """10 patients / 38 records have no diagnosis; NaN would hide that."""
        from ecgbench.labels.wctecgdb import load_labels

        header = self.HEADER.replace(self.DIAGNOSIS, "not reported")
        df = load_labels(
            self._tree(tmp_path, {"patient008/seg01": header}),
            self._config(sample_config),
        )
        row = df.loc["patient008_seg01"]

        assert row["diagnosis"] == "not reported"
        assert bool(row["diagnosis_reported"]) is False
        assert row["diagnosis_group"] == "Not reported"

    def test_reconstructed_precordials_are_flagged_per_record(
        self, tmp_path, sample_config
    ):
        """The 8 affected records are synthesised as V = UV - WCT, not measured."""
        from ecgbench.labels.wctecgdb import load_labels

        header = self.HEADER + "#Reconstruct Precordials: V1, V1-raw, V2, V2-raw\n"
        df = load_labels(
            self._tree(tmp_path, {"patient008/seg01": header}),
            self._config(sample_config),
        )
        row = df.loc["patient008_seg01"]

        assert row["reconstructed_precordials"] == ["V1", "V1-raw", "V2", "V2-raw"]
        assert bool(row["has_reconstructed_precordials"]) is True

    def test_unmapped_diagnosis_raises_because_it_is_the_split_label(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels.wctecgdb import load_labels

        header = self.HEADER.replace(self.DIAGNOSIS, "Brugada syndrome")
        with pytest.raises(ValueError, match="Unmapped diagnosis"):
            load_labels(
                self._tree(tmp_path, {"patient001/seg01": header}),
                self._config(sample_config),
            )

    def test_unexpected_sex_raises(self, tmp_path, sample_config):
        from ecgbench.labels.wctecgdb import load_labels

        header = self.HEADER.replace("#Sex: M", "#Sex: Male")
        with pytest.raises(ValueError, match="unexpected #Sex"):
            load_labels(
                self._tree(tmp_path, {"patient001/seg01": header}),
                self._config(sample_config),
            )

    def test_missing_comment_line_raises(self, tmp_path, sample_config):
        """All 540 headers carry all three; one missing means a truncated copy."""
        from ecgbench.labels.wctecgdb import load_labels

        header = "\n".join(
            line for line in self.HEADER.splitlines() if not line.startswith("#Age")
        ) + "\n"
        with pytest.raises(ValueError, match=r"missing the \['Age'\]"):
            load_labels(
                self._tree(tmp_path, {"patient001/seg01": header}),
                self._config(sample_config),
            )

    def test_header_listed_in_records_but_absent_names_the_release(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels.wctecgdb import load_labels

        (tmp_path / "RECORDS").write_text(
            "patient001/seg01\npatient001/seg02\n", encoding="utf-8"
        )
        (tmp_path / "patient001").mkdir()
        (tmp_path / "patient001/seg01.hea").write_text(self.HEADER, encoding="cp1252")

        with pytest.raises(LabelSourceMissingError, match="wctecgdb"):
            load_labels(tmp_path, self._config(sample_config))

    def test_every_group_in_the_map_is_one_of_the_eight(self):
        """The map is hand-written, so a typo would create a ninth silent class."""
        from ecgbench.labels.wctecgdb import DIAGNOSIS_GROUP

        assert len(DIAGNOSIS_GROUP) == 40  # 43 header strings, 40 after correction
        assert set(DIAGNOSIS_GROUP.values()) == {
            "Myocardial infarction",
            "Angina or coronary artery disease",
            "Atrial fibrillation or flutter",
            "Other tachyarrhythmia",
            "Cardiomyopathy or heart failure",
            "Bradyarrhythmia or conduction block",
            "Other or non-cardiac",
            "Not reported",
        }


class TestECGCIPALabels:
    """Drug exposure and interval measurements from four CDISC analysis datasets."""

    #: One adeg row per (record, parameter). The values are the real
    #: 00689D31-8491-4643-B3C8-45241FBBD47C measurements.
    INTERVALS = {
        "HR": 64.0,
        "RR": 943.333333,
        "PR": 188.0,
        "QRS": 78.0,
        "QT": 371.0,
        "QTCF": 378.284764,
        "JTP": 232.0,
        "JTPC": 239.98394,
        "TPTE": 61.0,
    }

    def _config(self, sample_config):
        from ecgbench.config import LabelConfig

        return replace(
            sample_config,
            slug="ecgcipa",
            record_id_column="record_id",
            patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"},
            default_sampling_rate=1000,
            url="https://physionet.org/content/ecgcipa/1.0.0/",
            labels=LabelConfig(source_csv="adeg.csv", join_column="EGREFID"),
        )

    def _adeg_rows(self, record, subject, *, replicate=2, params=None, **overrides):
        """One row per parameter, with the design context repeated as the source does."""
        params = self.INTERVALS if params is None else params
        base = {
            "STUDYID": "SCR-004",
            "USUBJID": subject,
            "TRTA": "Ranolazine",
            "TRTP": "Ranolazine",
            "TRTSEQA": "Ranolazine",
            "APERIOD": 1,
            "APERIODC": "Period 1",
            "ATPT": "54 hrs",
            "ATPTN": 24,
            "NRRLT": 6.0,
            "ARRLT": 5.928056,
            "ADTM": "2017-03-29 12:55:41",
            "ADY": 3,
            "APERDAY": 3,
            "AEGBLFL": None,
            "ECGPCFL": "Y",
            "EGREFID": record,
            "DTYPE": None,
            "BASE": None,
            "CHG": None,
        }
        base.update(overrides)
        return [
            {**base, "PARAMCD": code, "AVAL": value, "EGREPNUM": replicate}
            for code, value in params.items()
        ]

    def _tree(self, tmp_path, *, adeg_rows, records, adpc_rows=(), subjects=None):
        """Write RECORDS, the four analysis CSVs and empty raw/medians headers."""
        lines = [f"raw/{stem}" for stem in records]
        lines += [f"medians/{stem}" for stem in records]
        (tmp_path / "RECORDS").write_text("\n".join(lines) + "\n", encoding="utf-8")

        pd.DataFrame(adeg_rows).to_csv(tmp_path / "adeg.csv", index=False)
        pd.DataFrame(
            list(adpc_rows),
            columns=["USUBJID", "APERIOD", "ATPTN", "PARAMCD", "AVAL", "LLOQFL"],
        ).to_csv(tmp_path / "adpc.csv", index=False)

        subjects = subjects or [
            {
                "USUBJID": 1001,
                "AGE": 41,
                "SEX": "M",
                # Leading whitespace, exactly as the release ships it.
                "RACE": "  WHITE",
                "ETHNIC": "NOT HISPANIC OR LATINO",
                "ARM": "Ranolazine",
                "ACTARM": "Ranolazine",
            }
        ]
        pd.DataFrame(subjects).to_csv(tmp_path / "adsl.csv", index=False)
        pd.DataFrame(
            [
                {"USUBJID": s["USUBJID"], "PARAMCD": code, "AVAL": value}
                for s in subjects
                for code, value in (
                    ("HEIGHT", 180.0), ("WEIGHT", 84.0), ("BMI", 25.9),
                    ("SYSBP", 121.0), ("DIABP", 77.0),
                )
            ]
        ).to_csv(tmp_path / "addm.csv", index=False)
        return tmp_path

    def test_shipped_config_points_at_adeg(self):
        from ecgbench.config import load_config

        spec = load_config("ecgcipa").labels
        assert spec is not None and spec.available
        # The loader reads adpc/adsl/addm too, but adeg is the one it cannot do
        # without, so a missing-file error should name it.
        assert spec.source_csv == "adeg.csv"
        assert spec.join_column == "EGREFID"

    def test_paths_point_at_raw_and_medians_separately(self, tmp_path, sample_config):
        """signal_path must be the raw acquisition; medians/ is derived."""
        from ecgbench.labels.ecgcipa import load_labels as load

        record = "00689D31-8491-4643-B3C8-45241FBBD47C"
        path = self._tree(
            tmp_path,
            adeg_rows=self._adeg_rows(record, 1001),
            records=[f"1001/{record}"],
        )
        df = load(path, self._config(sample_config))

        assert df.index.name == "record_id"
        assert list(df.index) == [record]
        assert df.loc[record, "patient_id"] == "1001"
        assert df.loc[record, "signal_path"] == f"raw/1001/{record}"
        assert df.loc[record, "median_beat_path"] == f"medians/1001/{record}"

    def test_intervals_are_pivoted_out_of_the_long_table(self, tmp_path, sample_config):
        from ecgbench.labels.ecgcipa import load_labels as load

        record = "00689D31-8491-4643-B3C8-45241FBBD47C"
        path = self._tree(
            tmp_path,
            adeg_rows=self._adeg_rows(record, 1001),
            records=[f"1001/{record}"],
        )
        row = load(path, self._config(sample_config)).loc[record]

        assert row["hr_bpm"] == 64.0
        assert row["qt_ms"] == 371.0
        assert row["qtcf_ms"] == pytest.approx(378.284764)
        assert row["jtpeak_ms"] == 232.0
        assert row["tpeak_tend_ms"] == 61.0
        assert row["treatment"] == "Ranolazine"
        assert bool(row["has_matching_pk"]) is True
        assert bool(row["used_for_baseline"]) is False

    def test_the_two_time_axes_are_kept_apart(self, tmp_path, sample_config):
        """ATPT counts from the period's first dose, NRRLT from that day's dose.

        A record on study day 3 reads 54 and 6 for the same instant, so collapsing
        them onto one column would stack three dosing days on top of each other.
        """
        from ecgbench.labels.ecgcipa import load_labels as load

        record = "00689D31-8491-4643-B3C8-45241FBBD47C"
        path = self._tree(
            tmp_path,
            adeg_rows=self._adeg_rows(record, 1001),
            records=[f"1001/{record}"],
        )
        row = load(path, self._config(sample_config)).loc[record]

        assert row["timepoint"] == "54 hrs"
        assert row["nominal_hours_from_period_start"] == 54.0
        assert row["nominal_hours_from_reference"] == 6.0
        assert row["study_day"] == 3

    def test_missing_measurements_stay_missing(self, tmp_path, sample_config):
        """10 records have no PR and 9 no T intervals; a 0 would be a lie."""
        from ecgbench.labels.ecgcipa import load_labels as load

        record = "AAAAAAAA-0000-0000-0000-000000000001"
        no_t = {k: v for k, v in self.INTERVALS.items()
                if k in {"HR", "RR", "PR", "QRS"}}
        path = self._tree(
            tmp_path,
            adeg_rows=self._adeg_rows(record, 1001, params=no_t),
            records=[f"1001/{record}"],
        )
        row = load(path, self._config(sample_config)).loc[record]

        assert row["qrs_ms"] == 78.0
        for column in ("qt_ms", "qtcf_ms", "jtpeak_ms", "jtpeakc_ms", "tpeak_tend_ms"):
            assert pd.isna(row[column])

    def test_triplicate_average_rows_are_not_records(self, tmp_path, sample_config):
        """adeg's DTYPE=AVERAGE rows have a blank EGREFID and carry the endpoints.

        They are the only rows where BASE/CHG exist, so they must be excluded from
        the per-record frame and reachable separately — not silently dropped.
        """
        from ecgbench.labels.ecgcipa import load_labels as load
        from ecgbench.labels.ecgcipa import load_triplicate_averages

        record = "00689D31-8491-4643-B3C8-45241FBBD47C"
        averages = [
            {
                "STUDYID": "SCR-004", "USUBJID": 1001, "TRTA": "Ranolazine",
                "TRTP": "Ranolazine", "TRTSEQA": "Ranolazine", "APERIOD": 1,
                "APERIODC": "Period 1", "ATPT": "54 hrs", "ATPTN": 24,
                "NRRLT": 6.0, "ARRLT": 5.9, "ADTM": "2017-03-29 12:55:41",
                "ADY": 3, "APERDAY": 3, "AEGBLFL": None, "ECGPCFL": None,
                "EGREFID": None, "DTYPE": "AVERAGE", "PARAMCD": "QTCF",
                "AVAL": 400.0, "EGREPNUM": None, "BASE": 380.0, "CHG": 20.0,
            }
        ]
        config = self._config(sample_config)
        path = self._tree(
            tmp_path,
            adeg_rows=self._adeg_rows(record, 1001) + averages,
            records=[f"1001/{record}"],
        )

        df = load(path, config)
        assert len(df) == 1  # the AVERAGE row is not a record
        assert df.loc[record, "qtcf_ms"] == pytest.approx(378.284764)

        avg = load_triplicate_averages(path, config)
        assert len(avg) == 1
        assert avg.loc[0, "parameter"] == "QTCF"
        assert avg.loc[0, "CHG"] == 20.0
        # Keyed by subject/period/timepoint — there is no record to join to.
        assert {"patient_id", "period", "timepoint_n"} <= set(avg.columns)

    def test_replicate_number_is_anchored_and_disagreement_flagged(
        self, tmp_path, sample_config
    ):
        """EGREPNUM disagrees across parameters in 4 of the 5,749 records."""
        from ecgbench.labels.ecgcipa import load_labels as load

        record = "454D8FEA-5B86-4321-ABCD-FA4F8E1700E2"
        rows = self._adeg_rows(record, 1001, replicate=3)
        for row in rows:
            if row["PARAMCD"] in {"JTP", "JTPC", "QT", "QTCF", "TPTE"}:
                row["EGREPNUM"] = 2
        path = self._tree(tmp_path, adeg_rows=rows, records=[f"1001/{record}"])
        frame = load(path, self._config(sample_config))
        row = frame.loc[record]

        # Anchored on HR, which is measured for every record.
        assert row["replicate_number"] == 3
        assert bool(row["replicate_number_inconsistent"]) is True
        assert str(frame["replicate_number"].dtype) == "Int64"

    def test_dofetilide_concentration_keeps_its_own_unit(self, tmp_path, sample_config):
        """Dofetilide is pg/mL and the other six analytes are ng/mL."""
        from ecgbench.labels.ecgcipa import ANALYTE_COLUMNS

        assert ANALYTE_COLUMNS["DOF"] == ("plasma_dofetilide_pg_ml", "pg/mL")
        for code, (name, unit) in ANALYTE_COLUMNS.items():
            if code != "DOF":
                assert unit == "ng/mL"
                assert name.endswith("_ng_ml")

    def test_below_lloq_zeros_are_distinguishable_from_measurements(
        self, tmp_path, sample_config
    ):
        """263 adpc rows report 0 for 'below the limit of quantification'."""
        from ecgbench.labels.ecgcipa import load_labels as load

        measured = "AAAAAAAA-0000-0000-0000-00000000000A"
        censored = "AAAAAAAA-0000-0000-0000-00000000000B"
        path = self._tree(
            tmp_path,
            adeg_rows=(
                self._adeg_rows(measured, 1001, ATPTN=24)
                + self._adeg_rows(censored, 1001, ATPTN=5)
            ),
            records=[f"1001/{measured}", f"1001/{censored}"],
            adpc_rows=[
                (1001, 1, 24, "RAN", 4668.74, None),
                (1001, 1, 5, "RAN", 0.0, "Y"),
            ],
        )
        df = load(path, self._config(sample_config))

        assert df.loc[measured, "plasma_ranolazine_ng_ml"] == pytest.approx(4668.74)
        assert df.loc[measured, "plasma_below_lloq"] == ""
        assert bool(df.loc[measured, "plasma_any_below_lloq"]) is False
        # A 0 that means "censored", not "none detected".
        assert df.loc[censored, "plasma_ranolazine_ng_ml"] == 0.0
        assert df.loc[censored, "plasma_below_lloq"] == "RAN"
        assert bool(df.loc[censored, "plasma_any_below_lloq"]) is True

    def test_the_lloq_flag_survives_a_csv_round_trip(self, tmp_path, sample_config):
        """plasma_below_lloq is "" for uncensored records, which pandas reads back
        from a CSV as NaN — so `!= ""` on a re-read frame matches every row. The
        boolean is what the generated metadata CSV and any user filter rely on."""
        from ecgbench.labels.ecgcipa import load_labels as load

        measured = "AAAAAAAA-0000-0000-0000-00000000000A"
        path = self._tree(
            tmp_path,
            adeg_rows=self._adeg_rows(measured, 1001, ATPTN=24),
            records=[f"1001/{measured}"],
            adpc_rows=[(1001, 1, 24, "RAN", 4668.74, None)],
        )
        load(path, self._config(sample_config)).to_csv(tmp_path / "round_trip.csv")
        reread = pd.read_csv(tmp_path / "round_trip.csv")

        assert pd.isna(reread.loc[0, "plasma_below_lloq"])  # the trap
        assert reread.loc[0, "plasma_below_lloq"] != ""  # ... which reads as censored
        assert bool(reread.loc[0, "plasma_any_below_lloq"]) is False  # ... and this does not

    def test_race_whitespace_is_stripped(self, tmp_path, sample_config):
        """adsl ships '  WHITE' and ' ASIAN'; a groupby on those splits classes."""
        from ecgbench.labels.ecgcipa import load_labels as load

        record = "00689D31-8491-4643-B3C8-45241FBBD47C"
        path = self._tree(
            tmp_path,
            adeg_rows=self._adeg_rows(record, 1001),
            records=[f"1001/{record}"],
        )
        row = load(path, self._config(sample_config)).loc[record]

        assert row["race"] == "WHITE"
        assert row["age_years"] == 41
        assert row["bmi_kg_m2"] == 25.9
        assert row["systolic_bp_mmhg"] == 121.0

    def test_missing_adeg_names_the_file(self, tmp_path, sample_config):
        from ecgbench.labels.ecgcipa import load_labels as load

        (tmp_path / "RECORDS").write_text("raw/1001/x\nmedians/1001/x\n")
        with pytest.raises(LabelSourceMissingError, match="adeg.csv"):
            load(tmp_path, self._config(sample_config))

    def test_fiducials_are_resolved_positionally_not_by_symbol(self):
        """'(' marks both the P and QRS onsets, ')' both the QRS and T offsets.

        The real 00689D31-...D47C annotation, whose derived intervals equal
        adeg.csv's published PR 188, QRS 78, QT 371, J-Tpeak 232, Tpeak-Tend 61.
        """
        from ecgbench.labels.ecgcipa import _fiducials

        marks = [(165, "("), (353, "("), (396, "N"), (431, ")"), (663, "t"), (724, ")")]
        f = _fiducials(marks)

        assert f["p_onset_ms"] == 165
        assert f["qrs_onset_ms"] == 353
        assert f["qrs_peak_ms"] == 396
        assert f["qrs_offset_ms"] == 431
        assert f["t_peak_ms"] == 663
        assert f["t_offset_ms"] == 724
        assert f["t_peak_secondary_ms"] is None
        assert f["qrs_onset_ms"] - f["p_onset_ms"] == 188
        assert f["qrs_offset_ms"] - f["qrs_onset_ms"] == 78
        assert f["t_offset_ms"] - f["qrs_onset_ms"] == 371
        assert f["t_peak_ms"] - f["qrs_offset_ms"] == 232
        assert f["t_offset_ms"] - f["t_peak_ms"] == 61

    def test_fiducials_of_an_incompletely_annotated_beat(self):
        """9 records have no T annotation at all — the last ')' is then the QRS
        offset, and reading it as the T offset would invent a QT interval."""
        from ecgbench.labels.ecgcipa import _fiducials

        f = _fiducials([(353, "("), (396, "N"), (431, ")")])
        assert f["qrs_onset_ms"] == 353
        assert f["qrs_offset_ms"] == 431
        assert f["p_onset_ms"] is None  # only one onset: it is the QRS
        assert f["t_offset_ms"] is None
        assert f["t_peak_ms"] is None

        # 30 records carry a secondary T peak.
        g = _fiducials(
            [(165, "("), (353, "("), (396, "N"), (431, ")"),
             (600, "t"), (663, "t"), (724, ")")]
        )
        assert g["t_peak_ms"] == 600
        assert g["t_peak_secondary_ms"] == 663
        assert g["t_offset_ms"] == 724

    def test_every_interval_and_analyte_code_maps_to_a_unique_column(self):
        """Hand-written maps, so a duplicated value would silently overwrite."""
        from ecgbench.labels.ecgcipa import (
            ANALYTE_COLUMNS,
            INTERVAL_COLUMNS,
            VITAL_COLUMNS,
        )

        for mapping in (INTERVAL_COLUMNS, VITAL_COLUMNS):
            assert len(set(mapping.values())) == len(mapping)
        names = [name for name, _ in ANALYTE_COLUMNS.values()]
        assert len(set(names)) == len(names) == 7


class TestECGDMMLDLabels:
    """Drug exposure, intervals and morphology from one shipped clinical table."""

    #: Real values from 39BF8219-C83A-4121-926F-2BC730FBE127 (subject 2001,
    #: placebo, pre-dose), so the renames are checked against the release.
    ROW = {
        "EGREFID": "39BF8219-C83A-4121-926F-2BC730FBE127",
        "RANDID": 2001,
        "SEX": "M",
        "AGE": 21,
        "HGHT": 175.0,
        "WGHT": 60.3,
        "SYSBP": 125.0,
        "DIABP": 69.33,
        "RACE": "BLACK OR AFRICAN AMERICAN",
        "ETHNIC": "NOT HISPANIC OR LATINO",
        "ARMCD": "E-A-B-D-C",
        "VISIT": "PERIOD-1-DOSING",
        "TRTA": "Placebo",
        "DOF": None,
        "LIDO": None,
        "MEXI": None,
        "MOXI": None,
        "MOXI.M2": None,
        "DILT": None,
        "TPT": -0.5,
        "BASELINE": "Y",
        "RR": 1124.0,
        "PR": 166.0,
        "QT": 420.0,
        "QRS": 72.0,
        "JTPEAK": 263.0,
        "TPEAKTEND": 85.0,
        "TPEAKTPEAKP": None,
        "ERD_30": 52.0,
        "LRD_30": 28.0,
        "Twave_amplitude": 727.7845179035,
        "Twave_asymmetry": 0.1929824501,
        "Twave_flatness": 0.5300399661,
    }

    def _config(self, sample_config):
        from ecgbench.config import LabelConfig

        return replace(
            sample_config,
            slug="ecgdmmld",
            record_id_column="record_id",
            patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"},
            default_sampling_rate=1000,
            url="https://physionet.org/content/ecgdmmld/1.0.0/",
            labels=LabelConfig(
                source_csv="SCR-003.Clinical.Data.csv", join_column="EGREFID"
            ),
        )

    def _row(self, **overrides):
        return {**self.ROW, **overrides}

    def _tree(self, tmp_path, rows):
        pd.DataFrame(list(rows)).to_csv(
            tmp_path / "SCR-003.Clinical.Data.csv", index=False
        )
        return tmp_path

    def test_shipped_config_points_at_the_clinical_table(self):
        from ecgbench.config import load_config

        spec = load_config("ecgdmmld").labels
        assert spec is not None and spec.available
        assert spec.source_csv == "SCR-003.Clinical.Data.csv"
        assert spec.join_column == "EGREFID"

    def test_columns_are_renamed_and_indexed_by_record(self, tmp_path, sample_config):
        from ecgbench.labels.ecgdmmld import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))

        assert df.index.name == "record_id"
        assert list(df.index) == [self.ROW["EGREFID"]]
        row = df.iloc[0]
        assert row["patient_id"] == "2001"
        assert row["treatment"] == "Placebo"
        assert row["treatment_sequence"] == "E-A-B-D-C"
        assert row["timepoint_hours"] == -0.5
        assert row["rr_ms"] == 1124.0
        assert row["qt_ms"] == 420.0
        assert row["jtpeak_ms"] == 263.0
        assert row["erd_30_ms"] == 52.0
        # Microvolts, while the waveforms are millivolts.
        assert row["twave_amplitude_uv"] == pytest.approx(727.7845179035)

    def test_signal_paths_are_built_from_subject_and_record_id(
        self, tmp_path, sample_config
    ):
        """The release ships no path column at all; both are derived."""
        from ecgbench.labels.ecgdmmld import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))
        row = df.iloc[0]

        assert row["signal_path"] == f"raw/2001/{self.ROW['EGREFID']}"
        assert row["median_beat_path"] == f"medians/2001/{self.ROW['EGREFID']}"

    def test_period_is_parsed_from_the_visit_label(self, tmp_path, sample_config):
        from ecgbench.labels.ecgdmmld import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [
                    self._row(),
                    self._row(
                        EGREFID="r2",
                        VISIT="PERIOD-4-DOSING",
                        TRTA="Moxifloxacin + Diltiazem",
                    ),
                ],
            ),
            self._config(sample_config),
        )

        assert list(df["period"]) == [1, 4]
        assert str(df["period"].dtype) == "Int64"

    def test_baseline_flag_becomes_a_boolean(self, tmp_path, sample_config):
        from ecgbench.labels.ecgdmmld import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [self._row(), self._row(EGREFID="r2", BASELINE="N", TPT=2.0)],
            ),
            self._config(sample_config),
        )

        assert list(df["is_baseline"]) == [True, False]
        assert df["is_baseline"].dtype == bool

    def test_heart_rate_and_qtcf_are_derived_because_the_release_omits_them(
        self, tmp_path, sample_config
    ):
        """Neither exists in the source, and an uncorrected QT is not comparable."""
        from ecgbench.labels.ecgdmmld import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))
        row = df.iloc[0]

        assert row["hr_bpm"] == pytest.approx(60_000 / 1124.0)
        # Fridericia: QT / cbrt(RR seconds).
        assert row["qtcf_ms"] == pytest.approx(420.0 / (1.124 ** (1 / 3)))

    def test_jtpeakc_is_deliberately_not_derived(self, tmp_path, sample_config):
        """Rate-correcting J-Tpeak needs the study's own fitted exponent.

        Inventing one would produce a plausible column reproducing nobody's
        analysis, so the loader leaves it out — see the module docstring.
        """
        from ecgbench.labels.ecgdmmld import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))

        assert "jtpeakc_ms" not in df.columns
        assert "jtpeak_ms" in df.columns

    def test_tpeak_tpeakp_is_exposed_even_though_it_is_always_empty(
        self, tmp_path, sample_config
    ):
        """Documented by the release, NA in all 4,211 rows, and no .atr marks a
        secondary T peak. Exposed so its absence is visible, not inferred."""
        from ecgbench.labels.ecgdmmld import ALWAYS_EMPTY_COLUMNS
        from ecgbench.labels.ecgdmmld import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))

        assert "tpeak_tpeakp_ms" in ALWAYS_EMPTY_COLUMNS
        assert "tpeak_tpeakp_ms" in df.columns
        assert df["tpeak_tpeakp_ms"].isna().all()

    def test_dofetilide_keeps_its_own_unit_in_the_column_name(self):
        """Dofetilide is pg/mL and the other five are ng/mL; pooling is a 1000x
        error, so the unit cannot be lost in a rename."""
        from ecgbench.labels.ecgdmmld import ANALYTE_COLUMNS

        assert ANALYTE_COLUMNS["DOF"] == ("plasma_dofetilide_pg_ml", "pg/mL")
        for code, (name, unit) in ANALYTE_COLUMNS.items():
            if code != "DOF":
                assert unit == "ng/mL"
                assert name.endswith("_ng_ml")

    def test_median_beat_readable_flags_the_three_corrupt_headers(
        self, tmp_path, sample_config
    ):
        """Three v1.0.0 median headers name a nonexistent .dat for one channel, so
        wfdb.rdrecord raises. The raw/ records are fine."""
        from ecgbench.labels.ecgdmmld import MEDIAN_HEADER_CORRUPT
        from ecgbench.labels.ecgdmmld import load_labels as load

        broken = next(iter(MEDIAN_HEADER_CORRUPT))
        df = load(
            self._tree(tmp_path, [self._row(), self._row(EGREFID=broken)]),
            self._config(sample_config),
        )

        assert len(MEDIAN_HEADER_CORRUPT) == 3
        assert df.loc[self.ROW["EGREFID"], "median_beat_readable"]
        assert not df.loc[broken, "median_beat_readable"]

    def test_missing_source_file_names_it_and_says_where_to_get_it(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels import LabelSourceMissingError
        from ecgbench.labels.ecgdmmld import load_labels as load

        with pytest.raises(LabelSourceMissingError, match="SCR-003.Clinical.Data.csv"):
            load(tmp_path, self._config(sample_config))

    def test_a_missing_expected_column_raises_rather_than_producing_a_gap(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels.ecgdmmld import load_labels as load

        row = self._row()
        del row["QT"]
        with pytest.raises(ValueError, match="QT"):
            load(self._tree(tmp_path, [row]), self._config(sample_config))

    def test_baseline_deltas_reference_the_subject_period_pre_dose_mean(
        self, tmp_path, sample_config
    ):
        """The study's endpoint is change from baseline, and unlike ecgcipa it is
        computable per record here — the baseline is the pre-dose triplicate mean."""
        from ecgbench.labels.ecgdmmld import load_baseline_deltas

        rows = [
            self._row(EGREFID="b1", QT=400.0, RR=1000.0),
            self._row(EGREFID="b2", QT=410.0, RR=1000.0),
            self._row(EGREFID="b3", QT=420.0, RR=1000.0),
            self._row(EGREFID="p1", BASELINE="N", TPT=2.0, QT=450.0, RR=1000.0),
        ]
        df = load_baseline_deltas(
            self._tree(tmp_path, rows), self._config(sample_config)
        )

        # Baseline QT is the mean of the three pre-dose records: 410.
        assert df.loc["p1", "baseline_qt_ms"] == pytest.approx(410.0)
        assert df.loc["p1", "delta_qt_ms"] == pytest.approx(40.0)
        # The pre-dose records keep their own deviation from the triplicate mean.
        assert df.loc["b1", "delta_qt_ms"] == pytest.approx(-10.0)
        assert df.loc["b3", "delta_qt_ms"] == pytest.approx(10.0)

    def test_baseline_deltas_are_per_period_not_per_subject(
        self, tmp_path, sample_config
    ):
        """Each crossover period has its own pre-dose baseline; sharing one across
        periods would attribute a washout drift to the drug."""
        from ecgbench.labels.ecgdmmld import load_baseline_deltas

        rows = [
            self._row(EGREFID="p1b", VISIT="PERIOD-1-DOSING", QT=400.0),
            self._row(EGREFID="p2b", VISIT="PERIOD-2-DOSING", TRTA="Dofetilide", QT=440.0),
            self._row(
                EGREFID="p2x", VISIT="PERIOD-2-DOSING", TRTA="Dofetilide",
                BASELINE="N", TPT=2.0, QT=460.0,
            ),
        ]
        df = load_baseline_deltas(
            self._tree(tmp_path, rows), self._config(sample_config)
        )

        # Referenced against period 2's own 440, not period 1's 400.
        assert df.loc["p2x", "baseline_qt_ms"] == pytest.approx(440.0)
        assert df.loc["p2x", "delta_qt_ms"] == pytest.approx(20.0)

    def test_arm_sequence_disagreement_is_warned_about(
        self, tmp_path, sample_config, caplog
    ):
        """ARMCD indexed by period reproduces TRTA for all 4,211 records, so a
        disagreement means the arm codes or period numbering changed."""
        import logging

        from ecgbench.labels.ecgdmmld import load_labels as load

        # Sequence says period 1 is E (Placebo); claim Dofetilide instead.
        rows = [self._row(TRTA="Dofetilide")]
        with caplog.at_level(logging.WARNING, logger="ecgbench.labels.ecgdmmld"):
            load(self._tree(tmp_path, rows), self._config(sample_config))

        assert "treatment_sequence" in caplog.text

    def test_every_interval_and_analyte_code_maps_to_a_unique_column(self):
        """Hand-written maps, so a duplicated value would silently overwrite."""
        from ecgbench.labels.ecgdmmld import (
            ANALYTE_COLUMNS,
            CONTEXT_COLUMNS,
            INTERVAL_COLUMNS,
            MORPHOLOGY_COLUMNS,
            SUBJECT_COLUMNS,
        )

        for mapping in (
            INTERVAL_COLUMNS, MORPHOLOGY_COLUMNS, SUBJECT_COLUMNS, CONTEXT_COLUMNS,
        ):
            assert len(set(mapping.values())) == len(mapping)
        names = [name for name, _ in ANALYTE_COLUMNS.values()]
        assert len(set(names)) == len(names) == 6


class TestECGRDVQLabels:
    """SCR-002: single-agent crossover arms, a long-format PK table, and a
    32-bit wrap in two PR values."""

    #: Real values from 491af4aa-941a-4a89-b74c-b38d91cfc5e9 (subject 1001,
    #: ranolazine period, pre-dose), so the renames are checked against the release.
    ROW = {
        "EGREFID": "491af4aa-941a-4a89-b74c-b38d91cfc5e9",
        "RANDID": 1001,
        "SEX": "F",
        "AGE": 25,
        "HGHT": 161.5,
        "WGHT": 54.8,
        "SYSBP": 114.5,
        "DIABP": 64.25,
        "RACE": "WHITE",
        "ETHNIC": "NOT HISPANIC OR LATINO",
        "ARMCD": "A,C,E,D,B",
        "VISIT": "PERIOD-1-DOSING",
        "EXTRT": "Ranolazine",
        "EXDOSE": 1500.0,
        "EXDOSU": "mg",
        "TPT": -0.5,
        "BASELINE": "Y",
        "PCTEST": None,
        "PCSTRESN": None,
        "PCSTRESU": None,
        "RR": 902.0,
        "PR": 130.0,
        "QT": 400.0,
        "QRS": 95.0,
        "JTPEAK": 218.0,
        "TPEAKTEND": 87.0,
        "TPEAKTPEAKP": None,
        "ERD_30": 39.0,
        "LRD_30": 28.0,
        "Twave_amplitude": 688.131,
        "Twave_asymmetry": 0.104167,
        "Twave_flatness": 0.363812,
    }

    def _config(self, sample_config):
        from ecgbench.config import LabelConfig

        return replace(
            sample_config,
            slug="ecgrdvq",
            record_id_column="record_id",
            patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"},
            default_sampling_rate=1000,
            url="https://physionet.org/content/ecgrdvq/1.0.0/",
            labels=LabelConfig(
                source_csv="SCR-002.Clinical.Data.csv", join_column="EGREFID"
            ),
        )

    def _row(self, **overrides):
        return {**self.ROW, **overrides}

    def _tree(self, tmp_path, rows):
        pd.DataFrame(list(rows)).to_csv(
            tmp_path / "SCR-002.Clinical.Data.csv", index=False
        )
        return tmp_path

    def test_shipped_config_points_at_the_clinical_table(self):
        from ecgbench.config import load_config

        spec = load_config("ecgrdvq").labels
        assert spec is not None and spec.available
        assert spec.source_csv == "SCR-002.Clinical.Data.csv"
        assert spec.join_column == "EGREFID"

    def test_columns_are_renamed_and_indexed_by_record(self, tmp_path, sample_config):
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))

        assert df.index.name == "record_id"
        assert list(df.index) == [self.ROW["EGREFID"]]
        row = df.iloc[0]
        assert row["patient_id"] == "1001"
        assert row["treatment"] == "Ranolazine"
        # Comma-separated here; the sibling ecgdmmld uses dashes.
        assert row["treatment_sequence"] == "A,C,E,D,B"
        assert row["timepoint_hours"] == -0.5
        assert row["rr_ms"] == 902.0
        assert row["qt_ms"] == 400.0
        assert row["jtpeak_ms"] == 218.0
        assert row["erd_30_ms"] == 39.0
        # Microvolts, while the waveforms are millivolts.
        assert row["twave_amplitude_uv"] == pytest.approx(688.131)

    def test_signal_paths_are_built_from_subject_and_record_id(
        self, tmp_path, sample_config
    ):
        """The release ships no path column at all; both are derived."""
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))
        row = df.iloc[0]

        assert row["signal_path"] == f"raw/1001/{self.ROW['EGREFID']}"
        assert row["median_beat_path"] == f"medians/1001/{self.ROW['EGREFID']}"

    def test_period_is_parsed_from_the_visit_label(self, tmp_path, sample_config):
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [
                    self._row(),
                    self._row(
                        EGREFID="r2",
                        VISIT="PERIOD-4-DOSING",
                        EXTRT="Quinidine Sulph",
                        EXDOSE=400.0,
                    ),
                ],
            ),
            self._config(sample_config),
        )

        assert list(df["period"]) == [1, 4]
        assert str(df["period"].dtype) == "Int64"

    def test_baseline_flag_becomes_a_boolean(self, tmp_path, sample_config):
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [self._row(), self._row(EGREFID="r2", BASELINE="N", TPT=2.0)],
            ),
            self._config(sample_config),
        )

        assert list(df["is_baseline"]) == [True, False]
        assert df["is_baseline"].dtype == bool

    def test_heart_rate_and_qtcf_are_derived_because_the_release_omits_them(
        self, tmp_path, sample_config
    ):
        """Neither exists in the source, and an uncorrected QT is not comparable."""
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))
        row = df.iloc[0]

        assert row["hr_bpm"] == pytest.approx(60_000 / 902.0)
        # Fridericia: QT / cbrt(RR seconds).
        assert row["qtcf_ms"] == pytest.approx(400.0 / (0.902 ** (1 / 3)))

    def test_jtpeakc_is_deliberately_not_derived(self, tmp_path, sample_config):
        """Rate-correcting J-Tpeak needs the study's own fitted exponent."""
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(self._tree(tmp_path, [self._row()]), self._config(sample_config))

        assert "jtpeakc_ms" not in df.columns
        assert "jtpeak_ms" in df.columns

    def test_wrapped_pr_is_repaired_and_flagged(self, tmp_path, sample_config):
        """Two v1.0.0 records store PR as roughly -2^32, because the P onset fell
        before the start of the median-beat window and an unsigned subtraction
        wrapped. Adding 2^32 recovers the only physiologic residue."""
        from ecgbench.labels.ecgrdvq import PR_WRAP_MODULUS
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [
                    self._row(),
                    self._row(EGREFID="wrapped", PR=345.0 - PR_WRAP_MODULUS),
                ],
            ),
            self._config(sample_config),
        )

        assert df.loc["wrapped", "pr_ms"] == pytest.approx(345.0)
        assert bool(df.loc["wrapped", "pr_ms_repaired"]) is True
        # An ordinary PR is untouched and unflagged.
        assert df.loc[self.ROW["EGREFID"], "pr_ms"] == pytest.approx(130.0)
        assert bool(df.loc[self.ROW["EGREFID"], "pr_ms_repaired"]) is False

    def test_a_missing_pr_is_not_mistaken_for_a_wrap(self, tmp_path, sample_config):
        """9 records have no median beat and therefore no PR at all. NaN must stay
        NaN rather than being flagged as repaired."""
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(
            self._tree(tmp_path, [self._row(EGREFID="nomedian", PR=None)]),
            self._config(sample_config),
        )

        assert pd.isna(df.loc["nomedian", "pr_ms"])
        assert bool(df.loc["nomedian", "pr_ms_repaired"]) is False

    def test_plasma_concentration_is_long_format_with_its_unit(
        self, tmp_path, sample_config
    ):
        """One agent per period means one measurement per record, so the release
        names the analyte in a column instead of using six wide ones."""
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [
                    self._row(
                        EGREFID="ran", BASELINE="N", TPT=2.0,
                        PCTEST="Ranolazine", PCSTRESN=3725.0, PCSTRESU="ng/mL",
                    ),
                ],
            ),
            self._config(sample_config),
        )

        row = df.loc["ran"]
        assert row["plasma_analyte"] == "Ranolazine"
        assert row["plasma_concentration"] == pytest.approx(3725.0)
        assert row["plasma_concentration_unit"] == "ng/mL"

    def test_dofetilide_pg_ml_is_rescaled_into_the_ng_ml_column(
        self, tmp_path, sample_config
    ):
        """Dofetilide is reported in pg/mL and the other three analytes in ng/mL, so
        a mean over the raw column compares numbers 1000x apart in scale. The raw
        column and its unit are left exactly as shipped."""
        from ecgbench.labels.ecgrdvq import UNIT_NG_ML, UNIT_PG_ML
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [
                    self._row(
                        EGREFID="dof", EXTRT="Dofetilide", EXDOSE=500.0, EXDOSU="ug",
                        BASELINE="N", TPT=2.0,
                        PCTEST="Dofetilide", PCSTRESN=2790.0, PCSTRESU=UNIT_PG_ML,
                    ),
                    self._row(
                        EGREFID="ver", EXTRT="Verapamil HCL", EXDOSE=120.0,
                        BASELINE="N", TPT=2.0,
                        PCTEST="Verapamil", PCSTRESN=167.0, PCSTRESU=UNIT_NG_ML,
                    ),
                ],
            ),
            self._config(sample_config),
        )

        # Untouched.
        assert df.loc["dof", "plasma_concentration"] == pytest.approx(2790.0)
        # Rescaled: 2790 pg/mL is 2.79 ng/mL.
        assert df.loc["dof", "plasma_concentration_ng_ml"] == pytest.approx(2.79)
        # Already ng/mL, so unchanged.
        assert df.loc["ver", "plasma_concentration_ng_ml"] == pytest.approx(167.0)

    def test_an_unrecognised_concentration_unit_is_warned_about(
        self, tmp_path, sample_config, caplog
    ):
        """A reissue introducing a third unit must not be scaled by 1.0 in silence."""
        import logging

        from ecgbench.labels.ecgrdvq import load_labels as load

        rows = [self._row(PCTEST="Ranolazine", PCSTRESN=1.0, PCSTRESU="umol/L")]
        with caplog.at_level(logging.WARNING, logger="ecgbench.labels.ecgrdvq"):
            load(self._tree(tmp_path, rows), self._config(sample_config))

        assert "umol/L" in caplog.text

    def test_dose_carries_its_own_unit_because_they_differ(
        self, tmp_path, sample_config
    ):
        """500 for dofetilide is micrograms; 400-1500 for the rest are milligrams."""
        from ecgbench.labels.ecgrdvq import load_labels as load

        df = load(
            self._tree(
                tmp_path,
                [
                    self._row(),
                    self._row(
                        EGREFID="dof", EXTRT="Dofetilide",
                        EXDOSE=500.0, EXDOSU="ug",
                    ),
                ],
            ),
            self._config(sample_config),
        )

        assert df.loc[self.ROW["EGREFID"], "dose_unit"] == "mg"
        assert df.loc["dof", "dose_unit"] == "ug"

    def test_median_beat_available_flags_the_nine_records_without_one(
        self, tmp_path, sample_config
    ):
        """medians/ holds 5,223 of the 5,232 records. Every interval was measured
        from the median beat, so those 9 rows have no PR, QRS, QT or J-Tpeak — but
        their raw/ records are intact."""
        from ecgbench.labels.ecgrdvq import MEDIAN_BEAT_MISSING
        from ecgbench.labels.ecgrdvq import load_labels as load

        absent = next(iter(MEDIAN_BEAT_MISSING))
        df = load(
            self._tree(tmp_path, [self._row(), self._row(EGREFID=absent)]),
            self._config(sample_config),
        )

        assert len(MEDIAN_BEAT_MISSING) == 9
        assert df.loc[self.ROW["EGREFID"], "median_beat_available"]
        assert not df.loc[absent, "median_beat_available"]

    def test_missing_source_file_names_it_and_says_where_to_get_it(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels import LabelSourceMissingError
        from ecgbench.labels.ecgrdvq import load_labels as load

        with pytest.raises(LabelSourceMissingError, match="SCR-002.Clinical.Data.csv"):
            load(tmp_path, self._config(sample_config))

    def test_a_missing_expected_column_raises_rather_than_producing_a_gap(
        self, tmp_path, sample_config
    ):
        from ecgbench.labels.ecgrdvq import load_labels as load

        row = self._row()
        del row["QT"]
        with pytest.raises(ValueError, match="QT"):
            load(self._tree(tmp_path, [row]), self._config(sample_config))

    def test_baseline_deltas_reference_the_subject_period_pre_dose_mean(
        self, tmp_path, sample_config
    ):
        """All 109 (subject, period) pairs have their pre-dose triplicate, so the
        study's change-from-baseline endpoint is computable for every record."""
        from ecgbench.labels.ecgrdvq import load_baseline_deltas

        rows = [
            self._row(EGREFID="b1", QT=400.0, RR=1000.0),
            self._row(EGREFID="b2", QT=410.0, RR=1000.0),
            self._row(EGREFID="b3", QT=420.0, RR=1000.0),
            self._row(EGREFID="p1", BASELINE="N", TPT=2.0, QT=450.0, RR=1000.0),
        ]
        df = load_baseline_deltas(
            self._tree(tmp_path, rows), self._config(sample_config)
        )

        # Baseline QT is the mean of the three pre-dose records: 410.
        assert df.loc["p1", "baseline_qt_ms"] == pytest.approx(410.0)
        assert df.loc["p1", "delta_qt_ms"] == pytest.approx(40.0)
        # The pre-dose records keep their own deviation from the triplicate mean.
        assert df.loc["b1", "delta_qt_ms"] == pytest.approx(-10.0)
        assert df.loc["b3", "delta_qt_ms"] == pytest.approx(10.0)

    def test_baseline_deltas_are_per_period_not_per_subject(
        self, tmp_path, sample_config
    ):
        """Each crossover period has its own pre-dose baseline; sharing one across
        periods would attribute a washout drift to the drug."""
        from ecgbench.labels.ecgrdvq import load_baseline_deltas

        rows = [
            self._row(EGREFID="p1b", VISIT="PERIOD-1-DOSING", QT=400.0),
            self._row(
                EGREFID="p2b", VISIT="PERIOD-2-DOSING", EXTRT="Verapamil HCL",
                EXDOSE=120.0, QT=440.0,
            ),
            self._row(
                EGREFID="p2x", VISIT="PERIOD-2-DOSING", EXTRT="Verapamil HCL",
                EXDOSE=120.0, BASELINE="N", TPT=2.0, QT=460.0,
            ),
        ]
        df = load_baseline_deltas(
            self._tree(tmp_path, rows), self._config(sample_config)
        )

        # Referenced against period 2's own 440, not period 1's 400.
        assert df.loc["p2x", "baseline_qt_ms"] == pytest.approx(440.0)
        assert df.loc["p2x", "delta_qt_ms"] == pytest.approx(20.0)

    def test_arm_sequence_disagreement_is_warned_about(
        self, tmp_path, sample_config, caplog
    ):
        """ARMCD indexed by period reproduces EXTRT for all 5,232 records, so a
        disagreement means the arm codes or period numbering changed."""
        import logging

        from ecgbench.labels.ecgrdvq import load_labels as load

        # Sequence A,C,E,D,B says period 1 is A (Ranolazine); claim Placebo.
        rows = [self._row(EXTRT="Placebo")]
        with caplog.at_level(logging.WARNING, logger="ecgbench.labels.ecgrdvq"):
            load(self._tree(tmp_path, rows), self._config(sample_config))

        assert "treatment_sequence" in caplog.text

    def test_a_short_arm_sequence_is_not_a_disagreement(
        self, tmp_path, sample_config, caplog
    ):
        """Subject 1002 withdrew after 4 of the 5 periods and carries a 4-code
        sequence, so a period index past the end of it is expected."""
        import logging

        from ecgbench.labels.ecgrdvq import load_labels as load

        rows = [
            self._row(
                EGREFID="withdrawn", RANDID=1002, ARMCD="E,A,B,C",
                VISIT="PERIOD-4-DOSING", EXTRT="Verapamil HCL", EXDOSE=120.0,
            )
        ]
        with caplog.at_level(logging.WARNING, logger="ecgbench.labels.ecgrdvq"):
            load(self._tree(tmp_path, rows), self._config(sample_config))

        assert "treatment_sequence" not in caplog.text

    def test_every_column_map_is_injective(self):
        """Hand-written maps, so a duplicated value would silently overwrite."""
        from ecgbench.labels.ecgrdvq import (
            ANALYTE_COLUMNS,
            CONTEXT_COLUMNS,
            INTERVAL_COLUMNS,
            MORPHOLOGY_COLUMNS,
            SUBJECT_COLUMNS,
        )

        for mapping in (
            INTERVAL_COLUMNS, MORPHOLOGY_COLUMNS, SUBJECT_COLUMNS, CONTEXT_COLUMNS,
            ANALYTE_COLUMNS,
        ):
            assert len(set(mapping.values())) == len(mapping)

    def test_arm_codes_differ_from_the_sibling_release(self):
        """A,B,C,D,E mean different drugs in SCR-002 and SCR-003, so the two maps
        must never be shared. Both use E for placebo and nothing else agrees."""
        from ecgbench.labels.ecgdmmld import (
            ARM_CODE_TREATMENTS as DMMLD_CODES,
        )
        from ecgbench.labels.ecgrdvq import ARM_CODE_TREATMENTS as RDVQ_CODES

        assert RDVQ_CODES["E"] == DMMLD_CODES["E"] == "Placebo"
        for code in ("A", "B", "C", "D"):
            assert RDVQ_CODES[code] != DMMLD_CODES[code]


class TestECGRDVQFiducials:
    """Resolving median-beat annotation marks, including the two short patterns."""

    def test_the_usual_five_marks_resolve_in_order(self):
        """P onset, QRS onset, QRS offset, T peak, T offset — 5,175 of 5,223."""
        from ecgbench.labels.ecgrdvq import _fiducials

        # Real marks from 491af4aa-941a-4a89-b74c-b38d91cfc5e9.
        out = _fiducials([(171, "("), (301, "("), (396, ")"), (614, "t"), (701, ")")])

        assert out["p_onset_ms"] == 171
        assert out["qrs_onset_ms"] == 301
        assert out["qrs_offset_ms"] == 396
        assert out["t_peak_ms"] == 614
        assert out["t_offset_ms"] == 701
        assert out["t_peak_secondary_ms"] is None
        assert out["n_annotations"] == 5
        # And they reproduce the published intervals exactly.
        assert out["qrs_onset_ms"] - out["p_onset_ms"] == 130  # PR
        assert out["qrs_offset_ms"] - out["qrs_onset_ms"] == 95  # QRS
        assert out["t_offset_ms"] - out["qrs_onset_ms"] == 400  # QT
        assert out["t_peak_ms"] - out["qrs_offset_ms"] == 218  # J-Tpeak
        assert out["t_offset_ms"] - out["t_peak_ms"] == 87  # Tpeak-Tend

    def test_a_secondary_t_peak_is_captured(self):
        """42 records carry one, unlike the sibling ecgdmmld where no annotation
        marks a second T peak at all and TPEAKTPEAKP is empty in every row."""
        from ecgbench.labels.ecgrdvq import _fiducials

        # Real marks from 4d527f3e-7f0d-4daa-8ae6-a81af0619cdb (quinidine).
        out = _fiducials(
            [(137, "("), (304, "("), (404, ")"), (600, "t"), (672, "t"), (779, ")")]
        )

        assert out["t_peak_ms"] == 600
        assert out["t_peak_secondary_ms"] == 672
        # The published TPEAKTPEAKP for this record.
        assert out["t_peak_secondary_ms"] - out["t_peak_ms"] == 72
        # The T offset is still the last offset, not the one before the second peak.
        assert out["t_offset_ms"] == 779

    def test_a_missing_p_onset_leaves_pr_unrecoverable(self):
        """2 records (subject 1007, verapamil) have 4 marks and no P onset, because
        it lies before the start of the window — which is why their PR wrapped."""
        from ecgbench.labels.ecgrdvq import _fiducials

        # Real marks from c2017512-fefb-4058-9fd9-5a0950acc6a6.
        out = _fiducials([(311, "("), (404, ")"), (615, "t"), (698, ")")])

        assert out["p_onset_ms"] is None
        assert out["qrs_onset_ms"] == 311
        assert out["qrs_offset_ms"] == 404
        assert out["t_offset_ms"] == 698
        assert out["n_annotations"] == 4
        # Everything not needing the P onset still reproduces the published value.
        assert out["qrs_offset_ms"] - out["qrs_onset_ms"] == 93  # QRS
        assert out["t_offset_ms"] - out["qrs_onset_ms"] == 387  # QT

    def test_a_missing_t_offset_leaves_qt_unrecoverable(self):
        """4 records (subject 1004, quinidine) have no T offset — quinidine
        flattened the T wave until its end could not be marked."""
        from ecgbench.labels.ecgrdvq import _fiducials

        # Real marks from def367b2-2bc6-46e5-8b7c-c86a2e5513a4.
        out = _fiducials([(180, "("), (302, "("), (410, ")"), (685, "t")])

        assert out["p_onset_ms"] == 180
        assert out["qrs_onset_ms"] == 302
        assert out["qrs_offset_ms"] == 410
        assert out["t_peak_ms"] == 685
        assert out["t_offset_ms"] is None
        # PR, QRS and J-Tpeak survive; the published values.
        assert out["qrs_onset_ms"] - out["p_onset_ms"] == 122
        assert out["qrs_offset_ms"] - out["qrs_onset_ms"] == 108
        assert out["t_peak_ms"] - out["qrs_offset_ms"] == 275

    def test_no_annotation_file_yields_an_empty_row(self):
        """The 9 records with no median beat have no .atr to read."""
        from ecgbench.labels.ecgrdvq import _fiducials

        out = _fiducials([])

        assert out["n_annotations"] == 0
        assert all(
            out[key] is None
            for key in out
            if key != "n_annotations"
        )


class TestEyeTrackingECGLabels:
    """A reader study over ten ECG *images* — no waveforms, so no config."""

    #: Two stimuli whose real AOI grids differ, and whose suffixes are unrelated
    #: to their names ("Atrial fibrillation" -> AFib).
    NSR = "Normal sinus rhythm"
    VTACH = "Ventricular tachycardia"

    def _tree(self, tmp_path, table="grid"):
        """A stand-in for Datasets/<table>_Anonymized.csv, with the real quirks."""
        from ecgbench.labels.eye_tracking_ecg import AOI_TABLES

        rows = []

        def session(reader, stimulus, group, gender, age, labels):
            rows.append({
                "Study_name": "ECG Study", "Respondent_Name": reader,
                "Gender": gender, "Age": age, "Group": group, "Type": "Stimulus",
                "Label": stimulus, "Start": 1000, "Duration": 30000,
                "ParentStimulus": None, "Hit_time_G": None, "Fixations_Count": None,
                "First_Fixation_Duration": None, "Average_Fixations_Duration": None,
                "Respondent_ratio_G": None, "Time_spent_G_Percentage": None,
            })
            for i, (label, hit, fixations) in enumerate(labels):
                rows.append({
                    "Study_name": "ECG Study", "Respondent_Name": reader,
                    "Gender": gender, "Age": age, "Group": group,
                    "Type": "Static AOI", "Label": label, "Start": 1000,
                    "Duration": 30000, "ParentStimulus": stimulus,
                    "Hit_time_G": hit, "Fixations_Count": fixations,
                    "First_Fixation_Duration": -1 if fixations == 0 else 30,
                    "Average_Fixations_Duration": -1 if fixations == 0 else 40,
                    "Respondent_ratio_G": 0 if hit == -1 else 1,
                    "Time_spent_G_Percentage": float(i),
                })

        if table == "long_short":
            for reader, age in (("Consultant 1", 44), ("Med 1", 0)):
                for stimulus in (self.NSR, self.VTACH):
                    session(reader, stimulus, "Consultant", "MALE", age,
                            [("Long", 100, 5), ("Short", 200, 9)])
        else:
            for reader, age in (("Consultant 1", 44), ("Med 1", 0)):
                # A full-grid image: numbered limb leads, a strip, the footer.
                session(reader, self.NSR, "Consultant", "MALE", age, [
                    ("1 NSR", 500, 3),
                    ("aVR NSR", -1, 0),          # never gazed at
                    ("V5-3 NSR", 700, 2),
                    ("Information NSR", 900, 1),
                ])
                # VTach reuses one label for two distinct regions, as shipped.
                session(reader, self.VTACH, "Consultant", "MALE", age, [
                    ("1 VTach", 400, 4),
                    ("II-3 VTach", 800, 6),
                    ("II-3 VTach", 850, 1),
                ])

        path = tmp_path / AOI_TABLES[table]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path)
        return tmp_path

    # --------------------------------------------------------------- decoding

    def test_aoi_area_strips_the_per_image_suffix(self):
        """Labels are scoped per image, so 'V1 NSR' and 'V1 AFib' are one region."""
        from ecgbench.labels.eye_tracking_ecg import aoi_area

        assert aoi_area("V1 NSR", self.NSR) == "V1"
        assert aoi_area("V1 AFib", "Atrial fibrillation") == "V1"
        assert aoi_area("V5-3 NSR", self.NSR) == "V5-3"

    def test_aoi_area_strips_authoring_copy_suffixes(self):
        """Three complete-heart-block labels ship as '... copy copy copy'."""
        from ecgbench.labels.eye_tracking_ecg import aoi_area

        stimulus = "Complete heart block"
        assert aoi_area("II-2 CompleteHeartBlock copy", stimulus) == "II-2"
        assert aoi_area("II-4 CompleteHeartBlock copy copy copy", stimulus) == "II-4"

    def test_numbered_areas_are_the_limb_leads(self):
        """1/2/3 are leads I/II/III — the trap that hides them entirely."""
        from ecgbench.labels.eye_tracking_ecg import classify_area

        assert classify_area("1") == ("lead", "I")
        assert classify_area("2") == ("lead", "II")
        assert classify_area("3") == ("lead", "III")

    def test_classify_area_separates_strips_leads_and_the_footer(self):
        from ecgbench.labels.eye_tracking_ecg import classify_area

        assert classify_area("V5-3") == ("rhythm_strip", "V5")
        assert classify_area("II-4") == ("rhythm_strip", "II")
        assert classify_area("aVR") == ("lead", "aVR")
        assert classify_area("V3R") == ("lead", "V3R")
        # The footer is not a trace, so it has no lead.
        assert classify_area("Information") == ("information", None)

    def test_size_qualifiers_do_not_hide_a_strip_segment(self):
        """'V1-4 long' is a strip quarter; 'V1 short' is a lead box."""
        from ecgbench.labels.eye_tracking_ecg import classify_area

        assert classify_area("V1-4 long") == ("rhythm_strip", "V1")
        assert classify_area("V1 short") == ("lead", "V1")
        assert classify_area("V5 short") == ("lead", "V5")

    # ---------------------------------------------------------------- loading

    def test_derived_columns_make_cross_image_grouping_possible(self, tmp_path):
        from ecgbench.labels.eye_tracking_ecg import load_aoi_metrics

        df = load_aoi_metrics(self._tree(tmp_path))

        assert set(df.columns) >= {"aoi_area", "aoi_kind", "aoi_lead", "aoi_occurrence"}
        # The stimulus header rows are not AOI rows.
        assert (df.Type == "Static AOI").all()
        nsr = df[df.ParentStimulus == self.NSR]
        assert sorted(set(nsr.aoi_area)) == ["1", "Information", "V5-3", "aVR"]
        assert set(nsr.aoi_kind) == {"lead", "rhythm_strip", "information"}

    def test_reused_label_is_disambiguated_not_merged(self, tmp_path):
        """'II-3 VTach' names two regions; a naive key would collide."""
        from ecgbench.labels.eye_tracking_ecg import load_aoi_metrics

        df = load_aoi_metrics(self._tree(tmp_path))
        reused = df[df.Label == "II-3 VTach"]

        assert len(reused) == 4                      # 2 readers x 2 regions
        assert sorted(reused.aoi_occurrence) == [0, 0, 1, 1]
        # The two regions kept their own metrics rather than being averaged.
        first = reused[reused.aoi_occurrence == 0]
        second = reused[reused.aoi_occurrence == 1]
        assert set(first.Hit_time_G) == {800}
        assert set(second.Hit_time_G) == {850}
        # And the full key is unique, which is what lets callers index on it.
        key = ["Respondent_Name", "ParentStimulus", "Label", "aoi_occurrence"]
        assert not df.duplicated(key).any()

    def test_minus_one_sentinels_become_nan(self, tmp_path):
        """-1 means 'never happened'; averaged as -1 it silently skews everything."""
        from ecgbench.labels.eye_tracking_ecg import load_aoi_metrics

        df = load_aoi_metrics(self._tree(tmp_path))
        never_gazed = df[df.Label == "aVR NSR"]

        assert never_gazed.Hit_time_G.isna().all()
        assert never_gazed.First_Fixation_Duration.isna().all()
        assert never_gazed.Average_Fixations_Duration.isna().all()
        # Rows that did happen are untouched.
        assert df.loc[df.Label == "1 NSR", "Hit_time_G"].eq(500).all()

    def test_raw_sentinels_are_available_when_asked_for(self, tmp_path):
        from ecgbench.labels.eye_tracking_ecg import load_aoi_metrics

        df = load_aoi_metrics(self._tree(tmp_path), sentinels_to_nan=False)

        assert df.loc[df.Label == "aVR NSR", "Hit_time_G"].eq(-1).all()

    def test_zero_age_is_an_anonymisation_artefact_not_a_value(self, tmp_path):
        """54 of the 63 real readers carry Age 0, so notna() reports 100%."""
        from ecgbench.labels.eye_tracking_ecg import load_respondents

        readers = load_respondents(self._tree(tmp_path))

        assert len(readers) == 2
        assert readers.loc["Consultant 1", "Age"] == 44
        assert pd.isna(readers.loc["Med 1", "Age"])

    def test_sessions_are_the_stimulus_rows(self, tmp_path):
        from ecgbench.labels.eye_tracking_ecg import load_sessions

        sessions = load_sessions(self._tree(tmp_path))

        assert len(sessions) == 4                    # 2 readers x 2 images
        assert set(sessions.stimulus) == {self.NSR, self.VTACH}
        assert not sessions.duplicated(["Respondent_Name", "stimulus"]).any()

    def test_long_short_table_has_no_lead_structure(self, tmp_path):
        from ecgbench.labels.eye_tracking_ecg import load_aoi_metrics

        df = load_aoi_metrics(self._tree(tmp_path, "long_short"), table="long_short")

        assert set(df.aoi_area) == {"Long", "Short"}
        assert set(df.aoi_kind) == {"region"}
        assert df.aoi_lead.isna().all()

    def test_full_frame_attaches_the_session_and_the_image(self, tmp_path):
        from ecgbench.labels.eye_tracking_ecg import load_eye_tracking_ecg

        df = load_eye_tracking_ecg(self._tree(tmp_path))

        assert (df.session_duration_ms == 30000).all()
        assert df.loc[df.ParentStimulus == self.NSR, "stimulus_image"].eq(
            "ECGs/ECG_Images/Normal_Sinus_Rhythm.jpg"
        ).all()

    # ----------------------------------------------------------------- errors

    def test_missing_source_names_the_release(self, tmp_path):
        from ecgbench.labels.eye_tracking_ecg import load_aoi_metrics

        with pytest.raises(LabelSourceMissingError, match="eye-tracking-ecg"):
            load_aoi_metrics(tmp_path)

    def test_unknown_table_is_rejected(self, tmp_path):
        from ecgbench.labels.eye_tracking_ecg import load_aoi_metrics

        with pytest.raises(ValueError, match="table must be one of"):
            load_aoi_metrics(tmp_path, table="heatmap")

    def test_unknown_stimulus_is_rejected(self, tmp_path):
        """The image filenames are not derivable from the stimulus names."""
        from ecgbench.labels.eye_tracking_ecg import stimulus_image_path

        with pytest.raises(ValueError, match="Unknown stimulus"):
            stimulus_image_path(tmp_path, "STEMI")      # the CSV says "ST elevation MI"

    def test_every_stimulus_maps_to_an_image_and_a_suffix(self):
        """The three per-stimulus tables must agree on the same ten images."""
        from ecgbench.labels.eye_tracking_ecg import (
            STIMULUS_AOI_SUFFIXES,
            STIMULUS_IMAGES,
        )

        assert len(STIMULUS_IMAGES) == 10
        assert set(STIMULUS_IMAGES) == set(STIMULUS_AOI_SUFFIXES)

    def test_there_is_deliberately_no_eye_tracking_ecg_config(self):
        """A config would imply splitting ten pictures into ten folds."""
        from ecgbench.config import list_available_configs

        assert "eye_tracking_ecg" not in list_available_configs()
        assert "eye_tracking_dataset_for_12_lead_ecg_interpretation" not in (
            list_available_configs()
        )


class TestEchoNextLabels:
    """Echo-derived labels where a missing measurement reads as a negative flag."""

    def _table(self, tmp_path):
        """Rows covering the trap: measured-negative, measured-positive, unmeasured."""
        from ecgbench.labels.echonext import SOURCE_CSV

        pd.DataFrame({
            "ecg_key": [1, 2, 3, 4],
            "patient_key": ["p1", "p2", "p3", "p4"],
            "split": ["train", "train", "val", "test"],
            "age_at_ecg": [40, 55, 70, 90],
            "sex": ["male", "female", "male", "female"],
            # Measured and low / measured and high / never measured / never measured.
            "tr_max_velocity_value": [2.0, 4.0, None, None],
            "tr_max_gte_32_flag": [0, 1, 0, 0],
            "lvef_value": [60.0, 30.0, 55.0, None],
            "lvef_lte_45_flag": [0, 1, 0, 0],
            "aortic_stenosis_value": ["none", "severe", "presumed none", None],
            "aortic_stenosis_moderate_or_greater_flag": [0, 1, 0, 0],
            "shd_moderate_or_greater_flag": [0, 1, 0, 0],
        }).to_csv(tmp_path / SOURCE_CSV, index=False)
        return tmp_path

    def test_unmeasured_records_get_a_measured_mask(self, tmp_path):
        """A 0 flag means 'low' or 'never measured'; only the mask separates them."""
        from ecgbench.labels.echonext import load_labels

        df = load_labels(self._table(tmp_path))

        assert df.loc[1, "tr_max_gte_32_flag_measured"]      # measured, low
        assert df.loc[2, "tr_max_gte_32_flag_measured"]      # measured, high
        assert not df.loc[3, "tr_max_gte_32_flag_measured"]  # never measured
        assert not df.loc[4, "tr_max_gte_32_flag_measured"]
        # The flag itself is 0 for records 1, 3 and 4 alike — indistinguishable.
        assert df.loc[[1, 3, 4], "tr_max_gte_32_flag"].tolist() == [0, 0, 0]

    def test_prevalence_changes_once_unmeasured_records_are_masked(self, tmp_path):
        """The reason the mask exists: 25% of records vs 50% of measured ones."""
        from ecgbench.labels.echonext import load_labels

        df = load_labels(self._table(tmp_path))

        assert df["tr_max_gte_32_flag"].mean() == 0.25
        measured = df[df["tr_max_gte_32_flag_measured"]]
        assert measured["tr_max_gte_32_flag"].mean() == 0.5

    def test_the_composite_flag_gets_no_measured_mask(self, tmp_path):
        """It is a disjunction of the others, with no single measurement behind it."""
        from ecgbench.labels.echonext import COMPOSITE_FLAG, load_labels

        df = load_labels(self._table(tmp_path))

        assert COMPOSITE_FLAG in df.columns
        assert f"{COMPOSITE_FLAG}_measured" not in df.columns

    def test_presumed_none_stays_distinct_from_none(self, tmp_path):
        """It is the report parser inferring an absence, not measuring one."""
        from ecgbench.labels.echonext import load_labels

        df = load_labels(self._table(tmp_path))
        severity = df["aortic_stenosis_value"]

        assert str(severity.dtype) == "category"
        assert severity.cat.ordered
        assert severity.loc[1] == "none"
        assert severity.loc[3] == "presumed none"
        assert severity.loc[1] != severity.loc[3]
        # Ordered, so a >= comparison means what it looks like.
        assert severity.loc[2] > severity.loc[1]

    def test_indexed_by_record_id(self, tmp_path):
        from ecgbench.labels.echonext import JOIN_COLUMN, load_labels

        df = load_labels(self._table(tmp_path))

        assert df.index.name == JOIN_COLUMN
        assert df.index.is_unique
        assert len(df) == 4

    def test_missing_source_names_the_release(self, tmp_path):
        from ecgbench.labels.echonext import load_labels

        with pytest.raises(LabelSourceMissingError, match="physionet.org/content/echonext"):
            load_labels(tmp_path)

    # ------------------------------------------------- the README's column order

    def test_tabular_feature_order_is_not_the_readmes(self, tmp_path):
        """age_at_ecg is column 1 of the array, not column 6 as documented."""
        from ecgbench.labels.echonext import (
            README_TABULAR_FEATURE_COLUMNS,
            TABULAR_FEATURE_COLUMNS,
        )

        assert TABULAR_FEATURE_COLUMNS != README_TABULAR_FEATURE_COLUMNS
        assert TABULAR_FEATURE_COLUMNS.index("age_at_ecg") == 1
        assert README_TABULAR_FEATURE_COLUMNS.index("age_at_ecg") == 6
        # Same set, different order — this is a shift, not a different feature set.
        assert set(TABULAR_FEATURE_COLUMNS) == set(README_TABULAR_FEATURE_COLUMNS)

    def test_tabular_features_are_named_in_the_true_order(self, tmp_path):
        import numpy as np

        from ecgbench.labels.echonext import TABULAR_FEATURE_COLUMNS, load_tabular_features

        # Column j holds value j, so a mislabelled column is visible in the values.
        np.save(tmp_path / "EchoNext_val_tabular_features.npy",
                np.tile(np.arange(7, dtype=float), (3, 1)))

        df = load_tabular_features(tmp_path, "val")

        assert list(df.columns) == list(TABULAR_FEATURE_COLUMNS)
        assert df["age_at_ecg"].iloc[0] == 1.0        # column 1, per the true order
        assert df["qt_corrected"].iloc[0] == 6.0

    def test_unexpected_column_count_refuses_to_name_the_columns(self, tmp_path):
        """If a reissue changes the layout, the correction must not be applied blind."""
        import numpy as np

        from ecgbench.labels.echonext import load_tabular_features

        np.save(tmp_path / "EchoNext_val_tabular_features.npy", np.zeros((3, 9)))

        with pytest.raises(ValueError, match="re-verify"):
            load_tabular_features(tmp_path, "val")


class TestSymileMimicLabels:
    """Symile-MIMIC is a multimodal cohort over MIMIC-IV-ECG's records."""

    #: MIMIC-IV-ECG study_ids. 40870988 appears twice — 12 studies serve two
    #: admissions each in the real release — and the order is not ascending.
    STUDY_IDS = [46857043, 40870988, 40870988, 45321834, 46660648]
    HADM_IDS = [25296721, 22680060, 29914730, 26188372, 24043239]

    def _cohort(self, tmp_path, study_id=None, admittime=None):
        """A five-row stand-in for symile_mimic_data.csv."""
        from ecgbench.labels.symile_mimic import SOURCE_CSV

        paths = [f"files/p10/p1000000{i}/s{sid}/{sid}"
                 for i, sid in enumerate(self.STUDY_IDS)]
        frame = {
            "subject_id": [10000001, 14565909, 14565909, 10000004, 10000005],
            "hadm_id": self.HADM_IDS,
            "admittime": admittime or [
                "2186-11-29 03:56:00",
                # The duplicated study's later admission comes FIRST in file order,
                # so a "keep the first row" policy would pick the wrong one.
                "2133-12-19 21:00:00",
                "2133-12-19 16:14:00",
                "2150-08-20 16:32:00",
                "2165-01-04 23:01:00",
            ],
            "gender": ["F", "M", "M", "F", "M"],
            "age": [64, 52, 52, 71, 48],
            "anchor_age": [64, 52, 52, 71, 48],
            # The shipped `study_id` is the CXR's, duplicating cxr_study_id.
            "study_id": [54684191, 54684192, 54684193, 54684194, 54684195],
            "cxr_study_id": study_id or [54684191, 54684192, 54684193, 54684194, 54684195],
            "cxr_path": [f"files/p10/s{i}/{i}.jpg" for i in range(5)],
            "cxr_ViewPosition": ["PA", "AP", "AP", "AP", "PA"],
            "ecg_study_id": self.STUDY_IDS,
            "ecg_path": paths,
            "ecg_time": ["2186-11-28 23:16:00"] * 5,
            # CheXpert's four states: 1.0, 0.0, -1.0 uncertain, NaN not mentioned.
            "Atelectasis": [1.0, 0.0, -1.0, None, 1.0],
            "Cardiomegaly": [None, None, 1.0, 0.0, -1.0],
            "No Finding": [None, 1.0, None, None, None],
            "labs_all_nan": [0] * 5,
        }
        # All 50 labs, as every shipped file carries them. Two are given real
        # values — one well covered, one sparse — and the rest are empty.
        frame.update(self._lab_columns(percentiles=False))
        pd.DataFrame(frame).to_csv(tmp_path / SOURCE_CSV, index=False)
        return tmp_path

    def _lab_columns(self, percentiles):
        """The 50 lab columns (and optionally their percentile twins)."""
        from ecgbench.labels.symile_mimic import LABS, PERCENTILE_SUFFIX

        columns = {itemid: [None] * 5 for itemid in LABS}
        columns["51221"] = [35.4, 30.1, 31.0, None, 42.2]
        columns["50934"] = [None, None, 12.0, None, None]
        if percentiles:
            for itemid in LABS:
                columns[f"{itemid}{PERCENTILE_SUFFIX}"] = [0.6] * 5
            columns[f"51221{PERCENTILE_SUFFIX}"] = [0.7, 0.3, 0.4, 0.5, 0.9]
            columns[f"50934{PERCENTILE_SUFFIX}"] = [0.6, 0.6, 0.2, 0.6, 0.6]
        return columns

    def _split(self, tmp_path, split, retrieval=False):
        """A stand-in split CSV — no ecg_study_id column, as shipped."""
        from ecgbench.labels.symile_mimic import SPLIT_CSVS

        paths = [f"files/p10/p1000000{i}/s{sid}/{sid}"
                 for i, sid in enumerate(self.STUDY_IDS)]
        frame = {
            "subject_id": [10000001, 14565909, 14565909, 10000004, 10000005],
            "hadm_id": self.HADM_IDS,
            "cxr_path": [f"files/p10/s{i}/{i}.jpg" for i in range(5)],
            "Atelectasis": [1.0, 0.0, -1.0, None, 1.0],
            "ecg_path": paths,
            **self._lab_columns(percentiles=True),
        }
        if retrieval:
            # Two queries x 2 candidates, positives deliberately NOT first in file
            # order so the sort has something to do.
            frame = {k: [v[0], v[1], v[0], v[1]] for k, v in frame.items()}
            frame["hadm_id"] = [self.HADM_IDS[0], self.HADM_IDS[1]] * 2
            frame["label_hadm_id"] = [self.HADM_IDS[1], self.HADM_IDS[0],
                                      self.HADM_IDS[0], self.HADM_IDS[1]]
            frame["label"] = [0, 0, 1, 1]
        pd.DataFrame(frame).to_csv(tmp_path / SPLIT_CSVS[split], index=False)
        return tmp_path

    # -- the cohort table ----------------------------------------------------

    def test_indexed_by_the_admission_which_is_the_row_unit(self, tmp_path):
        """hadm_id is unique across all 11,622 rows; ecg_study_id is not."""
        from ecgbench.labels.symile_mimic import ROW_KEY, load_cohort

        df = load_cohort(self._cohort(tmp_path))

        assert df.index.name == ROW_KEY
        assert list(df.index) == self.HADM_IDS
        assert ROW_KEY not in df.columns
        assert df.loc[25296721, "ecg_study_id"] == 46857043

    def test_the_ambiguous_study_id_column_is_dropped(self, tmp_path):
        """The shipped 'study_id' is the CXR's; joining MIMIC on it matches nothing."""
        from ecgbench.labels.symile_mimic import load_cohort

        df = load_cohort(self._cohort(tmp_path))

        assert "study_id" not in df.columns          # the trap is gone
        assert "cxr_study_id" in df.columns          # nothing was lost
        assert "ecg_study_id" in df.columns          # the real key stays

    def test_a_disagreeing_study_id_is_kept_not_discarded(self, tmp_path, caplog):
        """Dropping it is only safe while it duplicates cxr_study_id exactly."""
        from ecgbench.labels.symile_mimic import load_cohort

        path = self._cohort(tmp_path, study_id=[1, 2, 3, 4, 5])
        with caplog.at_level("WARNING"):
            df = load_cohort(path)

        assert "study_id" in df.columns
        assert "disagree" in caplog.text

    def test_column_subset_keeps_the_twin_needed_to_drop_the_trap(self, tmp_path):
        from ecgbench.labels.symile_mimic import load_cohort

        df = load_cohort(self._cohort(tmp_path), columns=["study_id", "ecg_study_id"])

        # Asking for study_id yields the ECG key and no misleading column.
        assert list(df.columns) == ["ecg_study_id"]

    # -- keying by the host dataset's record id -------------------------------

    def test_by_study_id_indexes_by_the_host_key(self, tmp_path):
        from ecgbench.labels.symile_mimic import JOIN_COLUMN, by_study_id, load_cohort

        df = by_study_id(load_cohort(self._cohort(tmp_path)))

        assert df.index.name == JOIN_COLUMN
        assert df.index.is_unique
        assert "ecg_study_id" not in df.columns      # consumed by the index
        assert "hadm_id" in df.columns               # and nothing lost

    def test_duplicate_studies_resolve_to_the_earliest_admission(self, tmp_path):
        """Not the first row: the later admission comes first in file order here."""
        from ecgbench.labels.symile_mimic import by_study_id, load_cohort

        df = by_study_id(load_cohort(self._cohort(tmp_path)))

        assert len(df) == 4                                  # one of five dropped
        assert df.loc[40870988, "hadm_id"] == 29914730        # the 16:14 admission
        assert df.loc[40870988, "admittime"] == "2133-12-19 16:14:00"

    def test_duplicate_studies_fall_back_to_lowest_hadm_id(self, tmp_path):
        """Split CSVs drop admittime, so the tiebreak must still be deterministic."""
        from ecgbench.labels.symile_mimic import by_study_id, load_cohort

        df = load_cohort(self._cohort(tmp_path)).drop(columns="admittime")
        keyed = by_study_id(df)

        assert len(keyed) == 4
        assert keyed.loc[40870988, "hadm_id"] == 22680060      # the lower id

    def test_duplicate_studies_can_be_refused_instead(self, tmp_path):
        from ecgbench.labels.symile_mimic import by_study_id, load_cohort

        df = load_cohort(self._cohort(tmp_path))

        with pytest.raises(ValueError, match=r"more than one admission.*40870988"):
            by_study_id(df, on_duplicate="raise")
        # keep_all is the escape hatch for inspecting them.
        assert len(by_study_id(df, on_duplicate="keep_all")) == 5
        with pytest.raises(ValueError, match="on_duplicate must be"):
            by_study_id(df, on_duplicate="first")

    def test_prefix_reaches_the_demoted_row_key(self, tmp_path):
        """hadm_id was an index name, so add_prefix never touched it."""
        from ecgbench.labels.symile_mimic import by_study_id, load_cohort

        df = by_study_id(load_cohort(self._cohort(tmp_path), prefix="sym_"), prefix="sym_")

        assert "sym_hadm_id" in df.columns
        assert "hadm_id" not in df.columns
        assert df.index.name == "study_id"            # the key itself is not prefixed

    # -- the release's own splits --------------------------------------------

    def test_split_recovers_the_study_id_from_the_ecg_path(self, tmp_path):
        """Split CSVs drop ecg_study_id; the path's last segment is the record name."""
        from ecgbench.labels.symile_mimic import ECG_STUDY_COLUMN, load_split

        df = load_split(self._split(tmp_path, "train"), "train")

        assert list(df[ECG_STUDY_COLUMN]) == self.STUDY_IDS
        assert df.index.name == "hadm_id"

    def test_a_non_numeric_path_stem_refuses_to_guess(self, tmp_path):
        """A layout change must not silently produce keys that join to nothing."""
        from ecgbench.labels.symile_mimic import SPLIT_CSVS, load_split

        path = self._split(tmp_path, "train")
        frame = pd.read_csv(path / SPLIT_CSVS["train"])
        frame.loc[0, "ecg_path"] = "files/p10/p10000001/s46857043/46857043.dat"
        frame.to_csv(path / SPLIT_CSVS["train"], index=False)

        with pytest.raises(ValueError, match="do not end in a numeric"):
            load_split(path, "train")

    def test_retrieval_splits_group_candidates_positive_first(self, tmp_path):
        from ecgbench.labels.symile_mimic import load_split, retrieval_queries

        df = load_split(self._split(tmp_path, "test", retrieval=True), "test")

        assert list(df.index.names) == ["label_hadm_id", "hadm_id"]
        # Within each query the positive candidate sorts first.
        for query in df.index.get_level_values(0).unique():
            assert df.loc[query, "label"].iloc[0] == 1
        # One row per query, which is what the real 464 test admissions are.
        queries = retrieval_queries(df)
        assert len(queries) == 2
        assert queries.index.name == "hadm_id"
        assert set(queries["label"]) == {1}

    def test_unknown_split_is_refused(self, tmp_path):
        from ecgbench.labels.symile_mimic import load_split

        with pytest.raises(ValueError, match="split must be one of"):
            load_split(tmp_path, "validation")

    def test_missing_source_names_the_release(self, tmp_path):
        from ecgbench.labels.symile_mimic import load_cohort, load_split

        with pytest.raises(LabelSourceMissingError, match="symile-mimic"):
            load_cohort(tmp_path)
        with pytest.raises(LabelSourceMissingError, match="credentialed"):
            load_split(tmp_path, "train")

    # -- labs ----------------------------------------------------------------

    def test_labs_frame_kinds(self, tmp_path):
        from ecgbench.labels.symile_mimic import LABS, labs_frame, load_split

        df = load_split(self._split(tmp_path, "train"), "train")

        values = labs_frame(df, "value")
        assert values.shape == (5, len(LABS))
        assert values["51221"].iloc[0] == 35.4
        # Missing labs are genuine NaNs, so notna() is the missingness indicator.
        missing = labs_frame(df, "missingness")
        assert missing["51221"].tolist() == [1, 1, 1, 0, 1]
        assert missing["50934"].sum() == 1
        assert labs_frame(df, "percentile")["51221"].iloc[0] == 0.7
        # Names are opt-in; itemids are the default so the join key is visible.
        assert labs_frame(df, "value", names=True).columns[0] == LABS["51221"]

    def test_percentiles_are_absent_from_the_cohort_table(self, tmp_path):
        """They are derived per split from the train ECDF, so only split CSVs have them."""
        from ecgbench.labels.symile_mimic import labs_frame, load_cohort

        df = load_cohort(self._cohort(tmp_path))

        assert labs_frame(df, "value").shape[1] == 50
        with pytest.raises(ValueError, match="percentiles are derived per split"):
            labs_frame(df, "percentile")
        with pytest.raises(ValueError, match="kind must be one of"):
            labs_frame(df, "zscore")

    # -- CheXpert ------------------------------------------------------------

    def test_uncertain_is_not_swept_into_not_mentioned(self, tmp_path):
        """The bug this guards: fill NaN after mapping -1 to NaN loses `uncertain`."""
        from ecgbench.labels.symile_mimic import chexpert_targets, load_cohort

        df = load_cohort(self._cohort(tmp_path))
        targets = chexpert_targets(df)      # uncertain="nan", not_mentioned="negative"

        # Row 2 is -1.0 (uncertain) and must stay NaN; row 3 was NaN (not
        # mentioned) and becomes 0.0.
        assert pd.isna(targets["Atelectasis"].loc[29914730])
        assert targets["Atelectasis"].loc[26188372] == 0.0
        assert targets["Atelectasis"].loc[25296721] == 1.0
        assert int(targets["Atelectasis"].isna().sum()) == 1

    def test_both_ambiguous_states_are_choices(self, tmp_path):
        from ecgbench.labels.symile_mimic import chexpert_targets, load_cohort

        df = load_cohort(self._cohort(tmp_path))

        as_neg = chexpert_targets(df, uncertain="negative")
        assert as_neg["Atelectasis"].loc[29914730] == 0.0
        as_pos = chexpert_targets(df, uncertain="positive")
        assert as_pos["Atelectasis"].loc[29914730] == 1.0
        # keep/nan returns the four shipped states untouched.
        raw = chexpert_targets(df, uncertain="keep", not_mentioned="nan")
        assert raw["Atelectasis"].tolist()[:3] == [1.0, 0.0, -1.0]
        assert pd.isna(raw["Atelectasis"].loc[26188372])
        with pytest.raises(ValueError, match="uncertain must be"):
            chexpert_targets(df, uncertain="drop")
        with pytest.raises(ValueError, match="not_mentioned must be"):
            chexpert_targets(df, not_mentioned="positive")

    def test_only_the_findings_present_are_returned(self, tmp_path):
        """The cohort table has all 14; the split CSVs carry only 6."""
        from ecgbench.labels.symile_mimic import chexpert_targets, load_cohort, load_split

        cohort = chexpert_targets(load_cohort(self._cohort(tmp_path)))
        assert list(cohort.columns) == ["Atelectasis", "Cardiomegaly", "No Finding"]

        split = chexpert_targets(load_split(self._split(tmp_path, "train"), "train"))
        assert list(split.columns) == ["Atelectasis"]

    def test_a_frame_with_no_findings_says_where_they_live(self, tmp_path):
        from ecgbench.labels.symile_mimic import chexpert_targets, load_cohort

        df = load_cohort(self._cohort(tmp_path), columns=["ecg_study_id"])

        with pytest.raises(ValueError, match="load_cohort.. carries all 14"):
            chexpert_targets(df)

    def test_every_helper_accepts_the_prefix(self, tmp_path):
        """The prefix cannot be inferred: the 50 labs and their percentile twins
        share suffixes, so a suffix match on a prefixed frame is ambiguous."""
        from ecgbench.labels.symile_mimic import (
            chexpert_targets,
            labs_frame,
            load_split,
            retrieval_queries,
        )

        df = load_split(self._split(tmp_path, "train"), "train", prefix="sym_")

        assert labs_frame(df, "value", prefix="sym_").shape == (5, 50)
        assert chexpert_targets(df, prefix="sym_").shape == (5, 1)
        with pytest.raises(ValueError, match="prefix= matching"):
            labs_frame(df, "value")

        retr = load_split(self._split(tmp_path, "test", retrieval=True), "test",
                          prefix="sym_")
        assert len(retrieval_queries(retr, prefix="sym_")) == 2

    # -- the preprocessed tensors --------------------------------------------

    def _tensors(self, tmp_path, split="test"):
        directory = tmp_path / "data_npy" / split
        directory.mkdir(parents=True)
        # Value (i, lead) encodes the row and the lead, so a shifted or transposed
        # read cannot pass by accident.
        ecg = np.stack([
            np.tile(np.arange(12, dtype=np.float32) + i * 100, (5000, 1))[None]
            for i in range(4)
        ])
        np.save(directory / f"ecg_{split}.npy", ecg)
        np.save(directory / f"hadm_id_{split}.npy", np.array(self.HADM_IDS[:4]))
        if split == "test":
            np.save(directory / f"label_{split}.npy", np.array([1, 0, 1, 0]))
        return tmp_path

    def test_tensors_come_with_the_row_keys_they_are_aligned_to(self, tmp_path):
        from ecgbench.labels.symile_mimic import load_split_tensors

        array, index = load_split_tensors(self._tensors(tmp_path), "test", "ecg")

        assert array.shape == (4, 1, 5000, 12)
        assert list(index) == self.HADM_IDS[:4]
        assert index.name == "hadm_id"
        # Row 2 is the third record: its leads run 200..211.
        assert array[2, 0, 0, 0] == 200.0

    def test_leads_first_reorients_without_reordering_leads(self, tmp_path):
        from ecgbench.labels.symile_mimic import as_leads_first, load_split_tensors

        array, _ = load_split_tensors(self._tensors(tmp_path), "test", "ecg")

        batch = as_leads_first(array)
        assert batch.shape == (4, 12, 5000)
        # Lead 4 of record 0 is aVF in MIMIC-IV-ECG's order, value 4.
        assert batch[0, 4, 0] == 4.0
        assert as_leads_first(array[0]).shape == (12, 5000)
        with pytest.raises(ValueError, match="Expected"):
            as_leads_first(np.zeros((5000, 12)))

    def test_label_tensors_exist_for_retrieval_splits_only(self, tmp_path):
        from ecgbench.labels.symile_mimic import load_split_tensors

        path = self._tensors(tmp_path)
        labels, _ = load_split_tensors(path, "test", "label")
        assert labels.tolist() == [1, 0, 1, 0]

        self._tensors(tmp_path, "train")
        with pytest.raises(ValueError, match="not.*a retrieval split"):
            load_split_tensors(path, "train", "label")
        with pytest.raises(ValueError, match="modality must be one of"):
            load_split_tensors(path, "test", "waveform")

    def test_inconsistent_row_counts_are_refused(self, tmp_path):
        """A truncated copy must not produce a silently misaligned join."""
        from ecgbench.labels.symile_mimic import load_split_tensors

        path = self._tensors(tmp_path)
        np.save(path / "data_npy" / "test" / "hadm_id_test.npy", np.array(self.HADM_IDS))

        with pytest.raises(ValueError, match="alignment cannot be trusted"):
            load_split_tensors(path, "test", "ecg")

    # -- the rule this whole module exists to keep --------------------------

    def test_there_is_deliberately_no_symile_mimic_config(self):
        """A config would let `ecgbench splits` build a second partition of MIMIC-IV-ECG."""
        from ecgbench.config import list_available_configs

        available = list_available_configs()
        assert "symile_mimic" not in available
        assert "symile" not in available

    def test_it_is_not_registered_as_a_label_loader(self):
        """_custom_loaders maps *config* slugs to loaders, and there is no config."""
        from ecgbench.labels import _custom_loaders

        assert "symile_mimic" not in _custom_loaders()


class TestSTAFFIIILabels:
    """A wide per-patient spreadsheet, unpivoted to one row per record.

    STAFF III ships no per-record table at all: the annotations are one row per
    patient with up to five balloon inflations across 29 columns, and the record
    ids are unpadded file numbers ("7c") that must become record names ("007c").
    These fixtures rebuild that layout in miniature rather than touching real data.
    """

    # Positional layout of the real sheet: rows 0-9 are prose and merged
    # headings, data starts at row 10. Only the columns the loader reads are
    # filled in; the rest exist so the positional indices line up.
    N_COLUMNS = 29

    @staticmethod
    def _row(patient, age, sex, prior_mi, **cells):
        row = [None] * TestSTAFFIIILabels.N_COLUMNS
        row[0], row[1], row[2], row[28] = patient, age, sex, prior_mi
        for column, value in cells.items():
            row[int(column.lstrip("c"))] = value
        return row

    def _sheet(self, tmp_path, rows):
        """Write a spreadsheet with the real file's 10 leading non-data rows."""
        pytest.importorskip("openpyxl")
        preamble = [[None] * self.N_COLUMNS for _ in range(10)]
        frame = pd.DataFrame(preamble + rows)
        frame.to_excel(
            tmp_path / "STAFF-III-Database-Annotations.xlsx",
            header=False,
            index=False,
        )
        return tmp_path

    # -- the vessel mapping, which is pure ----------------------------------

    @pytest.mark.parametrize(
        ("artery", "territory"),
        [
            ("prox LAD", "LAD"),
            ("mid LAD", "LAD"),
            ("prox mid LAD", "LAD"),
            ("LAD diag", "LAD"),
            ("prox RCA", "RCA"),
            ("dist RCA", "RCA"),
            ("prox circ", "LCx"),
            ("mid circ", "LCx"),
            ("left main", "LM"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_artery_territory_covers_every_shipped_value(self, artery, territory):
        """All twelve free-text vessel strings in the release map somewhere."""
        from ecgbench.labels.staffiii import artery_territory

        assert artery_territory(artery) == territory

    def test_unrecognised_artery_becomes_unknown_not_a_wrong_territory(self):
        """A future release adding a vessel must surface, not be folded into LAD."""
        from ecgbench.labels.staffiii import UNKNOWN, artery_territory

        assert artery_territory("ramus intermedius") == UNKNOWN

    # -- header parsing ------------------------------------------------------

    HEADER = (
        "001a 9 1000 300000 20:26:00 27/09/1995\n"
        "001a.dat 16+512 1600 12 0 0 0 0  V1\n"
        "001a.dat 16+512 1600 12 0 0 0 0  V2\n"
        "# Age: 52\n"
        "# Sex: F\n"
    )

    def test_header_geometry_reads_length_rate_and_demographics(self, tmp_path):
        from ecgbench.labels.staffiii import read_header_geometry

        (tmp_path / "001a.hea").write_text(self.HEADER, encoding="utf-8")
        fields = read_header_geometry(tmp_path / "001a.hea")

        assert fields["n_samples"] == 300000
        assert fields["sampling_rate"] == 1000
        assert fields["header_age"] == "52"
        assert fields["header_sex"] == "F"

    def test_missing_age_comment_is_not_a_parse_failure(self, tmp_path):
        """Patients 14 and 15 have no '# Age:' line at all."""
        from ecgbench.labels.staffiii import read_header_geometry

        header = self.HEADER.replace("# Age: 52\n", "")
        (tmp_path / "014a.hea").write_text(header, encoding="utf-8")
        fields = read_header_geometry(tmp_path / "014a.hea")

        assert fields["header_age"] == ""  # empty, not None, so the column stays str
        assert fields["header_sex"] == "F"
        assert fields["n_samples"] == 300000

    # -- unpivoting the sheet ------------------------------------------------

    def test_file_numbers_are_zero_padded_into_record_names(self, tmp_path):
        """The sheet writes "7c"; the files are named "007c"."""
        from ecgbench.labels.staffiii import read_annotation_sheet

        root = self._sheet(
            tmp_path,
            [self._row(7, 66, "m", "no", c3="7a", c4="7b", c6="7c", c7="dist circ")],
        )
        entries = read_annotation_sheet(root)

        assert sorted(entries["record_name"]) == ["007a", "007b", "007c"]

    def test_one_row_per_inflation_not_per_record(self, tmp_path):
        """Nine real records hold two or three inflations; the sheet lists each."""
        from ecgbench.labels.staffiii import read_annotation_sheet

        root = self._sheet(
            tmp_path,
            [
                self._row(
                    7, 66, "m", "no",
                    c6="7c", c7="dist circ",
                    c10="7c", c11="prox circ",
                )
            ],
        )
        entries = read_annotation_sheet(root)
        inflations = entries[entries["recording_type"] == "BI"]

        assert len(inflations) == 2
        assert list(inflations["record_name"]) == ["007c", "007c"]
        assert list(inflations["recording_index"]) == [1, 2]
        assert sorted(inflations["occluded_artery"]) == ["dist circ", "prox circ"]

    def test_unused_patient_numbers_yield_no_records(self, tmp_path):
        """The sheet has 108 rows but 28, 67, 78 and 103 have no files."""
        from ecgbench.labels.staffiii import read_annotation_sheet

        root = self._sheet(
            tmp_path,
            [
                self._row(1, 52, "f", "no", c3="1a"),
                self._row(28, None, None, None),  # the grey line: no file numbers
            ],
        )
        entries = read_annotation_sheet(root)

        assert set(entries["patient"]) == {1}

    def test_question_mark_age_becomes_empty(self, tmp_path):
        """The sheet writes '?' for the two patients with no recorded age."""
        from ecgbench.labels.staffiii import read_patient_attributes

        root = self._sheet(tmp_path, [self._row(14, "?", "m", "no", c3="14a")])
        attributes = read_patient_attributes(root)

        assert attributes.loc[14, "age"] == ""
        assert attributes.loc[14, "sex"] == "M"
        # "no" is the sheet's own wording for absence, and must not read as missing.
        assert attributes.loc[14, "prior_mi_location"] == "no"
        assert attributes.loc[14, "prior_mi"] == "False"

    # -- stratification ------------------------------------------------------

    def _records(self, territories):
        """One BI record per patient, with the given primary territory."""
        return pd.DataFrame(
            {
                "record_name": [f"{i:03d}c" for i in range(1, len(territories) + 1)],
                "patient_number": list(range(1, len(territories) + 1)),
                "recording_type": ["BI"] * len(territories),
                "recording_index": ["1"] * len(territories),
                "artery_territory": territories,
            }
        )

    def test_stratify_class_is_the_patients_primary_territory(self):
        from ecgbench.labels.staffiii import attach_stratify_class

        df = attach_stratify_class(self._records(["LAD"] * 10 + ["RCA"] * 10))

        assert set(df["stratify_class"]) == {"LAD", "RCA"}
        assert df.loc[0, "primary_artery_territory"] == "LAD"

    def test_rare_territories_are_pooled_by_patient_count(self):
        """LM has 3 inflations across 2 patients in the real release."""
        from ecgbench.labels.staffiii import OTHER, attach_stratify_class

        df = attach_stratify_class(self._records(["LAD"] * 10 + ["RCA"] * 10 + ["LM"]))

        assert set(df["stratify_class"]) == {"LAD", "RCA", OTHER}
        # The unpooled territory stays available; only the fold label is pooled.
        assert df.loc[20, "primary_artery_territory"] == "LM"
        assert df.loc[20, "stratify_class"] == OTHER

    def test_multi_territory_patient_takes_its_first_inflation(self):
        """Ten patients had inflations in more than one territory."""
        from ecgbench.labels.staffiii import attach_stratify_class

        df = self._records(["LAD"] * 10 + ["RCA"] * 10)
        df.loc[0, "artery_territory"] = "LAD;LM"  # patient 1, two inflations
        out = attach_stratify_class(df)

        assert out.loc[0, "primary_artery_territory"] == "LAD"
        assert out.loc[0, "stratify_class"] == "LAD"

    # -- config wiring -------------------------------------------------------

    def test_labels_point_at_the_spreadsheet(self):
        from ecgbench.config import load_config

        spec = load_config("staffiii").labels
        assert spec is not None and spec.available
        assert spec.source_csv == "STAFF-III-Database-Annotations.xlsx"
        assert spec.join_column == "record_name"

    def test_it_is_registered_as_a_label_loader(self):
        """Without this the declarative loader would try read_csv on an .xlsx."""
        from ecgbench.labels import _custom_loaders

        assert "staffiii" in _custom_loaders()

    def test_missing_spreadsheet_names_the_file_and_where_to_get_it(self, tmp_path):
        from ecgbench.config import load_config
        from ecgbench.labels.staffiii import scan_records

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "001a.hea").write_text(self.HEADER, encoding="utf-8")
        load_config("staffiii")

        with pytest.raises(LabelSourceMissingError, match="STAFF-III-Database-Annotations"):
            scan_records(tmp_path)

    def test_scan_refuses_a_path_with_no_headers(self, tmp_path):
        """Pointing at the parent of the version directory is the usual mistake."""
        from ecgbench.labels.staffiii import scan_records

        with pytest.raises(LabelSourceMissingError, match="No NNNx.hea headers"):
            scan_records(tmp_path)


class TestCPSC2018Labels:
    """CPSC's nine classes, and the primary label the WFDB conversion destroyed."""

    def test_labels_come_from_headers_not_a_csv(self):
        """No metadata file ships — not even REFERENCE.csv — so the headers are it."""
        from ecgbench.config import load_config

        spec = load_config("cpsc_2018").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None
        assert spec.join_column == "record_name"

    def test_class_table_is_cpscs_own_nine_not_the_challenge_tables(self):
        """The mapping is closed at nine codes, with CPSC's names, not SNOMED's.

        The packaged Challenge 2020 table calls 164884008 "ventricular ectopics"
        and 426783006 "sinus rhythm"; CPSC calls them PVC and Normal. This is the
        CPSC dataset, so CPSC's vocabulary wins — but the codes must still be the
        challenge's, because they are what the shipped headers contain.
        """
        from ecgbench.labels.challenge2020 import load_dx_mapping
        from ecgbench.labels.cpsc_2018 import CPSC_CLASSES

        assert len(CPSC_CLASSES) == 9
        assert [i for i, _, _, _ in CPSC_CLASSES] == list(range(1, 10))
        assert len({c for _, c, _, _ in CPSC_CLASSES}) == 9
        assert len({a for _, _, a, _ in CPSC_CLASSES}) == 9

        by_abbr = {a: c for _, c, a, _ in CPSC_CLASSES}
        assert by_abbr["PVC"] == "164884008"
        assert by_abbr["NSR"] == "426783006"
        assert by_abbr["RBBB"] == "59118001"

        # Every code exists in the challenge table, under a different name.
        challenge = load_dx_mapping()
        for _, code, _, _ in CPSC_CLASSES:
            assert code in challenge.index
        assert challenge.loc["164884008", "abbreviation"] == "VEB"
        assert challenge.loc["426783006", "abbreviation"] == "NSR"

    def test_multi_label_records_expose_every_class(self):
        """476 of 6,877 records carry more than one class; dx must keep them all."""
        import pandas as pd

        from ecgbench.labels.cpsc_2018 import attach_dx_columns

        df = attach_dx_columns(pd.DataFrame({"dx": ["164889003,59118001", "426783006"]}))

        assert df.loc[0, "n_dx"] == 2
        assert df.loc[0, "dx_abbreviations"] == "AF,RBBB"
        assert df.loc[0, "dx_names"] == "Atrial fibrillation|Right bundle branch block"
        assert df.loc[0, "dx_class_indices"] == "2,5"
        assert df.loc[1, "n_dx"] == 1
        assert df.loc[1, "dx_abbreviations"] == "NSR"

    def test_shipped_dx_order_is_a_class_index_sort_not_a_primary_diagnosis(self):
        """A0043 is documented as First=5 (RBBB), Second=2 (AF); its header is AF-first.

        This is why the single-label reduction is named ``stratify_dx`` and not
        ``primary_dx``. If the first code were ever a diagnosis, this assertion
        would be the thing that noticed.
        """
        import pandas as pd

        from ecgbench.labels.cpsc_2018 import attach_dx_columns

        a0043 = attach_dx_columns(pd.DataFrame({"dx": ["164889003,59118001"]}))
        assert a0043.loc[0, "dx_abbreviations"].split(",")[0] == "AF"
        assert a0043.loc[0, "dx_class_indices"] == "2,5"  # ascending: it is a sort

    def test_stratify_reduction_takes_the_rarest_class(self):
        """The reduction is rarest-first, so the tail stays representable."""
        import pandas as pd

        from ecgbench.labels.cpsc_2018 import attach_dx_columns

        # RBBB (3 records) is commoner than STE (1), so STE wins for the record
        # carrying both.
        df = attach_dx_columns(
            pd.DataFrame({"dx": ["59118001,164931005", "59118001", "59118001"]})
        )
        assert df.loc[0, "stratify_dx_abbreviation"] == "STE"
        assert df.loc[1, "stratify_dx_abbreviation"] == "RBBB"

    def test_stratify_reduction_breaks_ties_deterministically(self):
        """Equally rare classes resolve to the lowest CPSC class index, not scan order."""
        import pandas as pd

        from ecgbench.labels.cpsc_2018 import attach_dx_columns

        forward = attach_dx_columns(pd.DataFrame({"dx": ["164889003,59118001"]}))
        reversed_ = attach_dx_columns(pd.DataFrame({"dx": ["59118001,164889003"]}))

        # AF is class 2, RBBB is class 5 — both appear once, so AF wins both ways.
        assert forward.loc[0, "stratify_dx"] == "164889003"
        assert reversed_.loc[0, "stratify_dx"] == "164889003"

    def test_unknown_codes_are_kept_rather_than_dropped(self):
        """The nine-class set is closed, so a tenth code must surface, not vanish."""
        import pandas as pd

        from ecgbench.labels.cpsc_2018 import UNMAPPED, attach_dx_columns

        df = attach_dx_columns(pd.DataFrame({"dx": ["164889003,999999999"]}))

        assert df.loc[0, "n_dx"] == 2
        assert df.loc[0, "dx_abbreviations"] == f"AF,{UNMAPPED}"
        # An unmapped code has no class index, so it drops out of that column only.
        assert df.loc[0, "dx_class_indices"] == "2"

    def test_age_sentinel_is_documented_and_not_silently_dropped(self):
        """-1 means nothing and must stay distinguishable from a genuinely absent age."""
        from ecgbench.labels.cpsc_2018 import AGE_SENTINELS

        assert AGE_SENTINELS == ("-1",)

    def test_missing_record_tree_names_the_directory_to_point_at(self):
        """The mirror is flat Training_WFDB/, which is easy to point past."""
        from pathlib import Path

        from ecgbench.config import load_config
        from ecgbench.labels import LabelSourceMissingError
        from ecgbench.labels.cpsc_2018 import RECORDS_DIR, scan_headers

        assert RECORDS_DIR == "Training_WFDB"
        with pytest.raises(LabelSourceMissingError, match=RECORDS_DIR):
            scan_headers(Path("/nonexistent/cpsc_2018"))

        # The config's own paths are relative to the same root.
        assert load_config("cpsc_2018").signal_path_columns == {500: "signal_path"}

    def test_variable_length_dataset_disables_the_truncation_check(self):
        """6 s to 144 s in 1,650 distinct lengths — expected_samples must stay empty."""
        from ecgbench.config import load_config

        assert load_config("cpsc_2018").validation.expected_samples == {}


class TestSPHLabels:
    """SPH's AHA statement grammar, which nothing else in ECGBench parses."""

    #: A minimal codebook: two primaries, one modifier.
    CODE_CSV = (
        "Category,Code,Description\n"
        "A,1,Normal ECG\n"
        "C,22,Sinus bradycardia\n"
        "F,60,Ventricular premature complex(es)\n"
        'D,31,"Atrial premature complexes, nonconducted"\n'
        "Modifier,310,Frequent\n"
    )

    def _tree(self, tmp_path, rows):
        """rows: list of (ecg_id, aha_code, patient_id, age, sex, n)."""
        (tmp_path / "code.csv").write_text(self.CODE_CSV, encoding="utf-8")
        lines = ["ECG_ID,AHA_Code,Patient_ID,Age,Sex,N,Date"]
        lines += [
            f"{rid},{code},{pid},{age},{sex},{n},2020-01-01"
            for rid, code, pid, age, sex, n in rows
        ]
        (tmp_path / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config, slug="sph", record_id_column="ecg_id",
            patient_id_column="patient_id", signal_path_columns={500: "signal_path"},
            default_sampling_rate=500, label_column="aha_primary_codes",
        )

    def test_labels_come_from_two_shipped_csvs_not_the_generated_cache(self):
        """Both sources ship, so labels do not depend on `ecgbench splits` first."""
        from ecgbench.config import load_config
        from ecgbench.labels.sph import CODE_CSV, METADATA_CSV

        spec = load_config("sph").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # module-based: needs the code.csv join
        assert spec.join_column == "ecg_id"
        assert (METADATA_CSV, CODE_CSV) == ("metadata.csv", "code.csv")

    def test_statement_grammar_splits_primaries_from_modifiers(self, tmp_path, sample_config):
        """';' separates statements, '+' attaches modifiers to one primary."""
        from ecgbench.labels.sph import load_labels as sph_labels

        root = self._tree(tmp_path, [("A00001", "60+310;22", "S1", 44, "M", 5000)])
        df = sph_labels(root, self._config(sample_config))
        row = df.loc["A00001"]

        assert row["n_statements"] == 2
        assert row["aha_primary_codes"] == "60;22"
        assert row["aha_primary_descriptions"] == (
            "Ventricular premature complex(es);Sinus bradycardia"
        )
        assert row["aha_primary_categories"] == "F;C"
        # The modifier is not promoted to a diagnosis.
        assert row["aha_modifier_codes"] == "310"
        assert row["aha_modifier_descriptions"] == "Frequent"
        # aha_statements keeps the attachment the two flat columns discard.
        assert row["aha_statements"] == "60+310;22"
        assert row["is_normal"] is False or not row["is_normal"]

    def test_repeated_normal_code_still_counts_as_normal(self, tmp_path, sample_config):
        """A02322 and A05000 ship as "1;1"; a string comparison would miss them."""
        from ecgbench.labels.sph import load_labels as sph_labels

        root = self._tree(
            tmp_path,
            [("A00001", "1", "S1", 40, "M", 5000), ("A00002", "1;1", "S2", 50, "F", 5000)],
        )
        df = sph_labels(root, self._config(sample_config))

        assert bool(df.loc["A00002", "is_normal"]) is True
        # Deduplicated: one primary code, but the raw statement count is still 2.
        assert df.loc["A00002", "n_primary_codes"] == 1
        assert df.loc["A00002", "n_statements"] == 2
        assert df.loc["A00002", "aha_primary_codes"] == "1"
        assert bool(df["is_normal"].all())

    def test_list_columns_use_semicolons_because_descriptions_contain_commas(
        self, tmp_path, sample_config
    ):
        """"Atrial premature complexes, nonconducted" cannot survive a comma join."""
        from ecgbench.labels.sph import LIST_SEPARATOR
        from ecgbench.labels.sph import load_labels as sph_labels

        assert LIST_SEPARATOR == ";"
        root = self._tree(tmp_path, [("A00001", "31;22", "S1", 44, "M", 5000)])
        df = sph_labels(root, self._config(sample_config))
        descriptions = df.loc["A00001", "aha_primary_descriptions"]

        assert "," in descriptions  # the comma is inside a description
        assert descriptions.split(LIST_SEPARATOR) == [
            "Atrial premature complexes, nonconducted",
            "Sinus bradycardia",
        ]

    def test_stratify_code_is_the_rarest_not_the_first(self, tmp_path, sample_config):
        """Rarest-first keeps the tail representable; first-listed does not."""
        from ecgbench.labels.sph import load_labels as sph_labels

        rows = [(f"A{i:05d}", "22", f"S{i}", 40, "M", 5000) for i in range(5)]
        rows.append(("A09999", "22;60", "S99", 40, "M", 5000))
        df = sph_labels(self._tree(tmp_path, rows), self._config(sample_config))

        # 22 occurs six times, 60 once, so the multi-label record goes to 60 even
        # though 22 is listed first.
        assert df.loc["A09999", "aha_primary_codes"] == "22;60"
        assert df.loc["A09999", "stratify_code"] == "60"
        assert df.loc["A09999", "stratify_description"] == "Ventricular premature complex(es)"
        assert df.loc["A00000", "stratify_code"] == "22"

    def test_length_comes_from_metadata_not_from_opening_a_signal_file(
        self, tmp_path, sample_config
    ):
        """N agrees with the arrays for all 25,770 records, so it is authoritative."""
        from ecgbench.labels.sph import SAMPLING_RATE
        from ecgbench.labels.sph import load_labels as sph_labels

        root = self._tree(
            tmp_path,
            [("A00001", "1", "S1", 40, "M", 5000), ("A00002", "1", "S2", 40, "M", 28000)],
        )
        df = sph_labels(root, self._config(sample_config))

        assert SAMPLING_RATE == 500
        assert list(df["n_samples"]) == [5000, 28000]
        assert list(df["duration_seconds"]) == [10.0, 56.0]
        # No signal files exist in this fixture at all, and the loader did not care.
        assert not (root / "records").exists()

    def test_signal_paths_are_relative_to_the_dataset_root(self, tmp_path, sample_config):
        """The release publishes no path column, so the loader invents one."""
        from ecgbench.labels.sph import RECORDS_DIR
        from ecgbench.labels.sph import load_labels as sph_labels

        root = self._tree(tmp_path, [("A00001", "1", "S1", 40, "M", 5000)])
        df = sph_labels(root, self._config(sample_config))

        assert RECORDS_DIR == "records"
        assert df.loc["A00001", "signal_path"] == "records/A00001.h5"

    def test_missing_metadata_and_codebook_each_name_their_file(self, tmp_path, sample_config):
        from ecgbench.labels.sph import load_code_table
        from ecgbench.labels.sph import load_labels as sph_labels

        with pytest.raises(LabelSourceMissingError, match="metadata.csv"):
            sph_labels(tmp_path, self._config(sample_config))

        # metadata.csv present but the codebook absent: a different message.
        (tmp_path / "metadata.csv").write_text(
            "ECG_ID,AHA_Code,Patient_ID,Age,Sex,N,Date\nA00001,1,S1,40,M,5000,2020-01-01\n",
            encoding="utf-8",
        )
        with pytest.raises(LabelSourceMissingError, match="code.csv"):
            sph_labels(tmp_path, self._config(sample_config))
        with pytest.raises(LabelSourceMissingError, match="code.csv"):
            load_code_table(tmp_path)

    def test_a_code_outside_the_codebook_surfaces_rather_than_vanishing(
        self, tmp_path, sample_config
    ):
        """The vocabulary is closed today; a future release must not fail silently."""
        from ecgbench.labels.sph import UNMAPPED
        from ecgbench.labels.sph import load_labels as sph_labels

        root = self._tree(tmp_path, [("A00001", "999;22", "S1", 40, "M", 5000)])
        df = sph_labels(root, self._config(sample_config))

        assert df.loc["A00001", "aha_primary_codes"] == "999;22"
        assert df.loc["A00001", "aha_primary_descriptions"] == f"{UNMAPPED};Sinus bradycardia"
        assert UNMAPPED in df.loc["A00001", "aha_primary_categories"]

    def test_variable_length_dataset_disables_the_truncation_check(self):
        """10 s to 56 s in 39 distinct lengths — expected_samples must stay empty."""
        from ecgbench.config import load_config

        assert load_config("sph").validation.expected_samples == {}


class TestNingboIVALabels:
    """The label is what the ablation proved, not what the ECG shows."""

    def _sheet(self, tmp_path, rows):
        """rows: list of (hospital_id, type, left_right, sublocation, gender)."""
        frame = pd.DataFrame(
            rows, columns=["HospitalID", "Type", "LeftRight", "Sublocation", "Gender"]
        )
        # .csv, not .xlsx — read_diagnosis prefers it and it needs no openpyxl.
        frame.to_csv(tmp_path / "Diagnosis.csv", index=False)
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config, slug="ningbo_iva", record_id_column="hospital_id",
            patient_id_column=None, signal_path_columns={2000: "signal_path"},
            default_sampling_rate=2000, label_column="left_right",
        )

    def test_labels_come_from_the_spreadsheet_not_the_generated_cache(self):
        """Diagnosis.xlsx ships, so labels do not depend on `ecgbench splits`."""
        from ecgbench.config import load_config
        from ecgbench.labels.ningbo_iva import DIAGNOSIS_FILES

        spec = load_config("ningbo_iva").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # pandas cannot read .xlsx declaratively
        assert spec.join_column == "hospital_id"
        assert DIAGNOSIS_FILES == ("Diagnosis.csv", "Diagnosis.xlsx")

    def test_right_left_are_renamed_to_the_papers_rvot_lvot(self, tmp_path, sample_config):
        from ecgbench.labels.ningbo_iva import load_labels as iva_labels

        root = self._sheet(
            tmp_path,
            [
                (1000364, "PVC", "Right", "AC", "female"),
                (991591, "VT", "Left", "LCC", "male"),
            ],
        )
        df = iva_labels(root, self._config(sample_config))

        assert list(df["left_right"]) == ["RVOT", "LVOT"]
        assert set(df.index) == {"1000364", "991591"}
        assert df.index.name == "hospital_id"

    def test_sex_keeps_the_shipped_spelling_and_adds_the_catalogue_one(
        self, tmp_path, sample_config
    ):
        """This is the only dataset spelling sex 'female'/'male' in lower case."""
        from ecgbench.labels.ningbo_iva import load_labels as iva_labels

        root = self._sheet(
            tmp_path,
            [(1, "PVC", "Right", "AC", "female"), (2, "PVC", "Left", "LCC", "male")],
        )
        df = iva_labels(root, self._config(sample_config))

        assert list(df["sex"]) == ["female", "male"]
        assert list(df["sex_code"]) == ["F", "M"]

    def test_blank_sublocations_are_left_blank_not_inferred(self, tmp_path, sample_config):
        """40 cells are blank and the paper's Table 2 explains them — but it is not us.

        The paper assigns 45 RVOT patients to "RVOTOther" where the spreadsheet has
        6 explicit plus 39 blanks. Filling them would put our inference into the
        published labels.
        """
        from ecgbench.labels.ningbo_iva import load_labels as iva_labels

        root = self._sheet(
            tmp_path,
            [(1, "PVC", "Right", "RVOTOther", "female"), (2, "PVC", "Right", None, "male")],
        )
        df = iva_labels(root, self._config(sample_config))

        assert df.loc["1", "sublocation"] == "RVOTOther"
        assert pd.isna(df.loc["2", "sublocation"])

    def test_both_signal_paths_are_exposed_but_only_the_raw_one_is_canonical(
        self, tmp_path, sample_config
    ):
        """The denoised copy is not sample-aligned with the raw one — see the module."""
        from ecgbench.config import load_config
        from ecgbench.labels.ningbo_iva import (
            DENOISED_DIR,
            SIGNAL_DIR,
        )
        from ecgbench.labels.ningbo_iva import (
            load_labels as iva_labels,
        )

        assert (SIGNAL_DIR, DENOISED_DIR) == ("PVCVTRawECGData", "PVCVTECGData")
        root = self._sheet(tmp_path, [(1000364, "PVC", "Right", "AC", "female")])
        df = iva_labels(root, self._config(sample_config))

        assert df.loc["1000364", "signal_path"] == "PVCVTRawECGData/1000364.csv"
        assert df.loc["1000364", "signal_path_denoised"] == "PVCVTECGData/1000364.csv"
        # Only the raw column is wired into the config, validated and exported.
        assert load_config("ningbo_iva").signal_path_columns == {2000: "signal_path"}

    def test_sampling_rate_is_2000_not_the_catalogue_default_500(self):
        """An EP-lab acquisition system, and the highest rate in the catalogue."""
        from ecgbench.config import load_config
        from ecgbench.labels.ningbo_iva import SAMPLING_RATE

        assert SAMPLING_RATE == 2000
        assert load_config("ningbo_iva").sampling_rates == [2000]

    def test_missing_spreadsheet_names_both_accepted_filenames(self, tmp_path, sample_config):
        from ecgbench.labels.ningbo_iva import load_labels as iva_labels

        with pytest.raises(LabelSourceMissingError, match="Diagnosis"):
            iva_labels(tmp_path, self._config(sample_config))

    def test_a_renamed_column_fails_loudly(self, tmp_path, sample_config):
        from ecgbench.labels.ningbo_iva import load_labels as iva_labels

        frame = pd.DataFrame(
            [(1, "PVC", "Right", "AC", "female")],
            columns=["HospitalID", "Type", "Tract", "Sublocation", "Gender"],
        )
        frame.to_csv(tmp_path / "Diagnosis.csv", index=False)
        with pytest.raises(ValueError, match="LeftRight"):
            iva_labels(tmp_path, self._config(sample_config))

    def test_variable_length_dataset_disables_the_truncation_check(self):
        """317 distinct lengths over 334 records — expected_samples must stay empty."""
        from ecgbench.config import load_config

        assert load_config("ningbo_iva").validation.expected_samples == {}


class TestCODE15Labels:
    """CODE-15%: six flags, and the trap that "no flag" is not "normal"."""

    COLUMNS = (
        "exam_id,age,is_male,nn_predicted_age,1dAVb,RBBB,LBBB,SB,ST,AF,"
        "patient_id,death,timey,normal_ecg,trace_file"
    )

    def _exams(self, tmp_path, rows):
        """rows: list of dicts overriding the defaults below."""
        lines = [self.COLUMNS]
        for row in rows:
            r = {
                "exam_id": 1, "age": 50, "is_male": "True", "nn_predicted_age": 51.0,
                "1dAVb": "False", "RBBB": "False", "LBBB": "False", "SB": "False",
                "ST": "False", "AF": "False", "patient_id": 100, "death": "False",
                "timey": 1.0, "normal_ecg": "False", "trace_file": "exams_part0.hdf5",
                **row,
            }
            lines.append(",".join(str(r[c]) for c in self.COLUMNS.split(",")))
        (tmp_path / "exams.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config, slug="code15", record_id_column="exam_id",
            patient_id_column="patient_id", signal_path_columns={400: "signal_path"},
            default_sampling_rate=400, label_column="abnormality_codes",
        )

    def test_the_six_flags_are_the_whole_vocabulary(self):
        from ecgbench.labels.code15 import ABNORMALITIES

        assert ABNORMALITIES == ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")

    def test_labels_come_from_the_one_shipped_csv(self):
        """exams.csv ships, so labels do not depend on `ecgbench splits` first."""
        from ecgbench.config import load_config
        from ecgbench.labels.code15 import EXAMS_CSV

        spec = load_config("code15").labels
        assert spec is not None and spec.available
        assert spec.source_csv is None  # module-based: needs the derived columns
        assert spec.join_column == "exam_id"
        assert EXAMS_CSV == "exams.csv"

    def test_no_abnormality_is_not_the_same_as_normal(self, tmp_path, sample_config):
        """173,347 real records are neither flagged nor normal — the central trap.

        A model trained on the six flags alone treats those as negatives for
        everything, which is wrong rather than merely uninformative, so the two
        cases must stay distinguishable.
        """
        from ecgbench.labels.code15 import load_labels as code15_labels

        root = self._exams(tmp_path, [
            {"exam_id": 1, "normal_ecg": "True"},                 # genuinely normal
            {"exam_id": 2, "normal_ecg": "False"},                # some other finding
            {"exam_id": 3, "RBBB": "True", "normal_ecg": "False"},
        ])
        df = code15_labels(root, self._config(sample_config))

        # Both unflagged records have an empty label list...
        assert df.loc[1, "abnormality_codes"] == ""
        assert df.loc[2, "abnormality_codes"] == ""
        # ...and are still told apart, by normal_ecg and by stratify_class.
        assert bool(df.loc[1, "normal_ecg"]) is True
        assert bool(df.loc[2, "normal_ecg"]) is False
        assert df.loc[1, "stratify_class"] == "NORMAL"
        assert df.loc[2, "stratify_class"] == "OTHER"
        assert df.loc[3, "stratify_class"] == "RBBB"

    def test_multi_label_records_join_their_codes_and_stratify_on_the_rarest(
        self, tmp_path, sample_config
    ):
        """Rarest-wins, computed from the frame rather than hardcoded."""
        from ecgbench.labels.code15 import load_labels as code15_labels

        # RBBB appears three times, LBBB once, so LBBB is the rarer of the pair.
        rows = [
            {"exam_id": 1, "RBBB": "True", "LBBB": "True"},
            {"exam_id": 2, "RBBB": "True"},
            {"exam_id": 3, "RBBB": "True"},
        ]
        df = code15_labels(self._exams(tmp_path, rows), self._config(sample_config))

        # Codes are listed in the declared column order, not the rarity order.
        assert df.loc[1, "abnormality_codes"] == "RBBB,LBBB"
        assert df.loc[1, "n_abnormalities"] == 2
        # But the single-label reduction takes the rarer one.
        assert df.loc[1, "stratify_class"] == "LBBB"
        assert df.loc[2, "stratify_class"] == "RBBB"

    def test_absent_mortality_followup_stays_absent(self, tmp_path, sample_config):
        """112,132 real records have no follow-up; reading NaN as False invents
        112,132 survivors."""
        from ecgbench.labels.code15 import load_labels as code15_labels

        root = self._exams(tmp_path, [
            {"exam_id": 1, "death": "True", "timey": 2.5},
            {"exam_id": 2, "death": "False", "timey": 3.0},
            {"exam_id": 3, "death": "", "timey": ""},          # no follow-up
        ])
        df = code15_labels(root, self._config(sample_config))

        assert df["death"].dtype == "boolean"          # nullable, not plain bool
        assert bool(df.loc[1, "death"]) is True
        assert bool(df.loc[2, "death"]) is False
        assert pd.isna(df.loc[3, "death"])
        assert list(df["has_followup"]) == [True, True, False]
        # The false reading this guards against.
        assert int(df["death"].fillna(False).sum()) == 1

    def test_sex_is_derived_from_is_male_and_both_are_kept(self, tmp_path, sample_config):
        from ecgbench.labels.code15 import load_labels as code15_labels

        root = self._exams(tmp_path, [
            {"exam_id": 1, "is_male": "True"}, {"exam_id": 2, "is_male": "False"},
        ])
        df = code15_labels(root, self._config(sample_config))

        assert list(df["sex"]) == ["M", "F"]
        assert list(df["is_male"]) == [True, False]

    def test_a_renamed_column_fails_loudly(self, tmp_path, sample_config):
        from ecgbench.labels.code15 import load_labels as code15_labels

        (tmp_path / "exams.csv").write_text("exam_id,age\n1,50\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            code15_labels(tmp_path, self._config(sample_config))

    def test_missing_exams_csv_says_where_to_get_it(self, tmp_path, sample_config):
        from ecgbench.labels.code15 import load_labels as code15_labels

        with pytest.raises(LabelSourceMissingError, match="exams.csv"):
            code15_labels(tmp_path, self._config(sample_config))


class TestCODETestLabels:
    """CODE-test: eight keyless files joined by row position, seven annotators."""

    ABN = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")

    def _tree(self, tmp_path, n=4, flags=None, annotator_flags=None):
        """Build a data/ tree of `n` rows. `flags` sets the gold standard."""
        (tmp_path / "annotations").mkdir(exist_ok=True)
        attributes = ["age,sex"] + [f"{30 + i},{'MF'[i % 2]}" for i in range(n)]
        (tmp_path / "attributes.csv").write_text(
            "\n".join(attributes) + "\n", encoding="utf-8"
        )

        from ecgbench.labels.code_test import ANNOTATORS

        for name in ANNOTATORS:
            rows = (annotator_flags or {}).get(name) or flags or [()] * n
            header = ("," if name == "dnn" else "") + ",".join(self.ABN)
            lines = [header]
            for i, on in enumerate(rows):
                cells = ["1" if code in on else "0" for code in self.ABN]
                lines.append(",".join(([str(i)] if name == "dnn" else []) + cells))
            (tmp_path / "annotations" / f"{name}.csv").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config, slug="code_test", record_id_column="record_id",
            patient_id_column=None, signal_path_columns={400: "signal_path"},
            default_sampling_rate=400, label_column="abnormality_codes",
        )

    def _patch_n(self, monkeypatch, n):
        """The real loader hard-codes 827; shrink it for the fixtures."""
        import ecgbench.labels.code_test as mod

        monkeypatch.setattr(mod, "N_RECORDS", n)

    def test_the_record_id_is_the_row_index(self, tmp_path, sample_config, monkeypatch):
        """No identifier ships, so position is the only key there is."""
        from ecgbench.labels.code_test import load_labels as ct_labels

        self._patch_n(monkeypatch, 4)
        df = ct_labels(self._tree(tmp_path, 4), self._config(sample_config))

        assert list(df.index) == [0, 1, 2, 3]
        assert df.index.name == "record_id"
        assert list(df["age"]) == [30, 31, 32, 33]
        assert list(df["sex"]) == ["M", "F", "M", "F"]

    def test_all_seven_annotator_sets_are_exposed_side_by_side(
        self, tmp_path, sample_config, monkeypatch
    ):
        """Reader agreement is what this release is for; only the gold standard
        would throw that away."""
        from ecgbench.labels.code_test import ANNOTATORS
        from ecgbench.labels.code_test import load_labels as ct_labels

        self._patch_n(monkeypatch, 2)
        root = self._tree(
            tmp_path, 2,
            flags=[("AF",), ()],
            annotator_flags={"medical_students": [("AF", "RBBB"), ("SB",)]},
        )
        df = ct_labels(root, self._config(sample_config))

        assert len(ANNOTATORS) == 7
        for name in ANNOTATORS:
            assert f"{name}_AF" in df.columns
            assert f"{name}_abnormality_codes" in df.columns

        # The disagreement survives into the frame rather than being resolved.
        assert bool(df.loc[0, "gold_standard_AF"]) is True
        assert bool(df.loc[0, "medical_students_RBBB"]) is True
        assert bool(df.loc[0, "gold_standard_RBBB"]) is False
        assert df.loc[1, "medical_students_abnormality_codes"] == "SB"
        assert df.loc[1, "gold_standard_abnormality_codes"] == ""

    def test_the_unprefixed_columns_are_the_gold_standard(
        self, tmp_path, sample_config, monkeypatch
    ):
        from ecgbench.labels.code_test import GOLD_STANDARD
        from ecgbench.labels.code_test import load_labels as ct_labels

        self._patch_n(monkeypatch, 2)
        root = self._tree(
            tmp_path, 2,
            flags=[("LBBB",), ()],
            annotator_flags={"dnn": [("ST",), ("ST",)]},
        )
        df = ct_labels(root, self._config(sample_config))

        assert GOLD_STANDARD == "gold_standard"
        assert df.loc[0, "abnormality_codes"] == "LBBB"
        assert bool(df.loc[0, "LBBB"]) is True
        # Not the DNN's reading, even though it also has an opinion.
        assert bool(df.loc[0, "ST"]) is False
        assert bool(df.loc[0, "dnn_ST"]) is True

    def test_dnn_csv_extra_index_column_is_dropped(
        self, tmp_path, sample_config, monkeypatch
    ):
        """dnn.csv alone carries a leading unnamed column; the other six do not."""
        from ecgbench.labels.code_test import load_annotations

        self._patch_n(monkeypatch, 3)
        root = self._tree(tmp_path, 3, flags=[("AF",), (), ("SB",)])
        annotations = load_annotations(root)

        assert list(annotations["dnn"].columns) == list(self.ABN)
        assert not any(c.startswith("Unnamed") for c in annotations["dnn"].columns)
        assert list(annotations["dnn"].index) == [0, 1, 2]

    def test_a_source_file_of_the_wrong_length_is_refused_not_mis_joined(
        self, tmp_path, sample_config, monkeypatch
    ):
        """A positional join cannot partially match — it silently mislabels."""
        from ecgbench.labels.code_test import load_labels as ct_labels

        self._patch_n(monkeypatch, 4)
        root = self._tree(tmp_path, 4)
        # One annotator file loses a row.
        path = root / "annotations" / "cardiologist2.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="rows, expected 4"):
            ct_labels(root, self._config(sample_config))

    def test_stratify_takes_the_rarest_gold_standard_class(
        self, tmp_path, sample_config, monkeypatch
    ):
        """AF is 13 of 827 in the real release; a commoner class winning would
        leave folds with no AF at all."""
        from ecgbench.labels.code_test import load_labels as ct_labels

        self._patch_n(monkeypatch, 4)
        root = self._tree(tmp_path, 4, flags=[("RBBB", "AF"), ("RBBB",), ("RBBB",), ()])
        df = ct_labels(root, self._config(sample_config))

        assert df.loc[0, "abnormality_codes"] == "RBBB,AF"
        assert df.loc[0, "stratify_class"] == "AF"      # the rarer of the two
        assert df.loc[1, "stratify_class"] == "RBBB"
        # Not "NORMAL": this release publishes no normal flag at all.
        assert df.loc[3, "stratify_class"] == "NONE"

    def test_missing_annotations_point_at_the_data_subdirectory(
        self, tmp_path, sample_config
    ):
        """The archive extracts to data/, which is the commonest --data-path slip."""
        from ecgbench.labels.code_test import load_labels as ct_labels

        with pytest.raises(LabelSourceMissingError, match="data/"):
            ct_labels(tmp_path, self._config(sample_config))


class TestSamiTropLabels:
    """SaMi-Trop: no diagnoses, complete mortality follow-up, positional join."""

    COLUMNS = "exam_id,age,is_male,normal_ecg,death,timey,nn_predicted_age"

    def _exams(self, tmp_path, rows):
        lines = [self.COLUMNS]
        for row in rows:
            r = {
                "exam_id": 1, "age": 60, "is_male": "True", "normal_ecg": "False",
                "death": "False", "timey": 2.0, "nn_predicted_age": 61.0, **row,
            }
            lines.append(",".join(str(r[c]) for c in self.COLUMNS.split(",")))
        (tmp_path / "exams.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config, slug="sami_trop", record_id_column="exam_id",
            patient_id_column=None, label_column="stratify_class",
        )

    def test_the_source_is_named_so_the_manifest_can_checksum_it(self):
        from ecgbench.config import load_config
        from ecgbench.labels.sami_trop import EXAMS_CSV

        spec = load_config("sami_trop").labels
        assert spec is not None and spec.available
        assert spec.source_csv == EXAMS_CSV == "exams.csv"
        assert spec.join_column == "exam_id"

    def test_row_position_is_exposed_because_it_is_the_join(
        self, tmp_path, sample_config, monkeypatch
    ):
        """exams.hdf5 carries no identifier, so the row number IS the key.

        It is returned as a column rather than left implicit, because the
        splitter builds ``exams.hdf5:tracings:<row>`` from it.
        """
        import ecgbench.labels.sami_trop as mod

        monkeypatch.setattr(mod, "N_RECORDS", 3)
        root = self._exams(tmp_path, [{"exam_id": 77}, {"exam_id": 11}, {"exam_id": 42}])
        df = mod.load_labels(root, self._config(sample_config))

        # Not sorted: the row is the CSV's own order, which is what the waveform
        # array uses. Sorting the frame here would corrupt every signal path.
        assert list(df.index) == [77, 11, 42]
        assert list(df["row"]) == [0, 1, 2]

    def test_a_wrong_row_count_is_refused_rather_than_partially_joined(
        self, tmp_path, sample_config
    ):
        """A positional join against a file of the wrong length mislabels everything.

        There is no partial-match state to warn about — every row after the first
        discrepancy is confidently wrong — so this must raise.
        """
        from ecgbench.labels.sami_trop import load_labels as sami_labels

        root = self._exams(tmp_path, [{"exam_id": 1}, {"exam_id": 2}])
        with pytest.raises(ValueError, match="row position|1631"):
            sami_labels(root, self._config(sample_config))

    def test_mortality_comes_first_in_the_stratification(
        self, tmp_path, sample_config, monkeypatch
    ):
        """Death wins over normal_ecg, so the 3 dead-and-normal records go to DEATH.

        The alternative — a death x normal_ecg cross — has a cell of 3 records in
        the real release, which cannot be spread over 10 folds at all.
        """
        import ecgbench.labels.sami_trop as mod

        monkeypatch.setattr(mod, "N_RECORDS", 4)
        root = self._exams(tmp_path, [
            {"exam_id": 1, "death": "True", "normal_ecg": "True"},
            {"exam_id": 2, "death": "True", "normal_ecg": "False"},
            {"exam_id": 3, "death": "False", "normal_ecg": "True"},
            {"exam_id": 4, "death": "False", "normal_ecg": "False"},
        ])
        df = mod.load_labels(root, self._config(sample_config))

        assert list(df["stratify_class"]) == [
            mod.STRATIFY_DEATH,      # dead AND normal -> DEATH, not a 4th class
            mod.STRATIFY_DEATH,
            mod.STRATIFY_NORMAL,
            mod.STRATIFY_ABNORMAL_ALIVE,
        ]

    def test_follow_up_is_complete_so_death_is_a_plain_bool(
        self, tmp_path, sample_config, monkeypatch
    ):
        """Unlike CODE-15%, every SaMi-Trop record has an outcome.

        CODE-15% needs a nullable boolean and a has_followup flag because 112,132
        of its records have neither. Here all 1,631 do, so the simpler type is
        correct and a NaN would be a bug rather than a third state.
        """
        import ecgbench.labels.sami_trop as mod

        monkeypatch.setattr(mod, "N_RECORDS", 2)
        root = self._exams(tmp_path, [{"exam_id": 1}, {"exam_id": 2, "death": "True"}])
        df = mod.load_labels(root, self._config(sample_config))

        assert df["death"].dtype == bool
        assert df["death"].notna().all()
        assert "has_followup" not in df.columns


class TestIKEMLabels:
    """IKEM: -1 means missing in every numeric column, and no diagnoses ship."""

    COLUMNS = (
        "exam_id,acquisition_date,patient_id,age,is_male,weight,height,"
        "ventricular_rate,atrial_rate"
    )

    def _exams(self, tmp_path, rows):
        lines = [self.COLUMNS]
        for row in rows:
            r = {
                "exam_id": 1, "acquisition_date": "03-21-2016", "patient_id": "abc",
                "age": 66, "is_male": 1, "weight": -1, "height": -1,
                "ventricular_rate": 70, "atrial_rate": 70, **row,
            }
            lines.append(",".join(str(r[c]) for c in self.COLUMNS.split(",")))
        (tmp_path / "exams.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config, slug="ikem", record_id_column="exam_id",
            patient_id_column="patient_id", label_column="stratify_class",
        )

    def test_this_release_ships_no_diagnoses(self, tmp_path, sample_config):
        """The paper's diagnostic labels are not part of the Zenodo release.

        Worth an explicit test: the dataset looks like a classification corpus —
        98,130 hospital ECGs — and is not one. Nothing in the loader may invent a
        diagnosis column.
        """
        from ecgbench.labels.ikem import load_labels as ikem_labels

        root = self._exams(tmp_path, [{"exam_id": 1}])
        df = ikem_labels(root, self._config(sample_config))
        assert not [c for c in df.columns if "diagnos" in c.lower() or c == "dx"]

    def test_minus_one_becomes_nan_in_every_numeric_column(self, tmp_path, sample_config):
        """Read literally, IKEM's mean weight is about -76 kg.

        89.6% of weights and 89.3% of heights are -1, so this is the difference
        between a usable column and a wrong one — and ``notna()`` reports the
        source as 100% complete either way.
        """
        from ecgbench.labels.ikem import SENTINEL_COLUMNS
        from ecgbench.labels.ikem import load_labels as ikem_labels

        root = self._exams(tmp_path, [
            {"exam_id": 1, "age": -1, "weight": -1, "height": -1,
             "ventricular_rate": -1, "atrial_rate": -1, "is_male": -1},
            {"exam_id": 2, "age": 40, "weight": 70, "height": 175,
             "ventricular_rate": 80, "atrial_rate": 80, "is_male": 0},
        ])
        df = ikem_labels(root, self._config(sample_config))

        for column in SENTINEL_COLUMNS:
            assert column in df.columns
            assert not (df[column].astype(float) == -1).any(), column
        first = df.loc[1]
        assert pd.isna(first["age"]) and pd.isna(first["weight"])
        assert pd.isna(first["ventricular_rate"])
        # Unknown sex stays unknown rather than becoming female.
        assert pd.isna(first["is_male"]) and pd.isna(first["sex"])
        assert df.loc[2, "sex"] == "F"
        assert list(df["has_weight"]) == [False, True]

    def test_the_date_is_month_first_and_parsing_it_wrongly_is_silent(
        self, tmp_path, sample_config
    ):
        """MM-DD-YYYY. Day-first inference would swap the first 12 days of a month.

        It also sorts wrongly as a string: string min/max over the real column
        reports 2018-2021 where the true range is 2004-2022.
        """
        from ecgbench.labels.ikem import load_labels as ikem_labels

        root = self._exams(tmp_path, [
            {"exam_id": 1, "acquisition_date": "03-21-2016"},
            {"exam_id": 2, "acquisition_date": "07-04-2019"},
        ])
        df = ikem_labels(root, self._config(sample_config))

        assert str(df.loc[1, "acquisition_date"].date()) == "2016-03-21"
        # 07-04 is 4 July, not 7 April.
        assert str(df.loc[2, "acquisition_date"].date()) == "2019-07-04"
        assert list(df["acquisition_year"]) == [2016, 2019]

    def test_rate_bands_have_three_classes_and_absorb_the_unmeasurable(
        self, tmp_path, sample_config
    ):
        """Only 7 real records have no usable rate, which is too few for a class.

        A 7-member class cannot be spread over 10 folds, so those records join
        NORMAL. A rate band is a measurement, not a rhythm diagnosis.
        """
        from ecgbench.labels.ikem import (
            STRATIFY_BRADY,
            STRATIFY_NORMAL,
            STRATIFY_TACHY,
        )
        from ecgbench.labels.ikem import load_labels as ikem_labels

        root = self._exams(tmp_path, [
            {"exam_id": 1, "ventricular_rate": 45},
            {"exam_id": 2, "ventricular_rate": 75},
            {"exam_id": 3, "ventricular_rate": 140},
            {"exam_id": 4, "ventricular_rate": -1},
            {"exam_id": 5, "ventricular_rate": 0},
        ])
        df = ikem_labels(root, self._config(sample_config))

        assert list(df["stratify_class"]) == [
            STRATIFY_BRADY, STRATIFY_NORMAL, STRATIFY_TACHY,
            STRATIFY_NORMAL,  # sentinel
            STRATIFY_NORMAL,  # a literal 0 bpm is not bradycardia, it is unmeasured
        ]

    def test_real_lengths_are_optional_so_metadata_only_copies_still_load(
        self, tmp_path, sample_config
    ):
        """The true pre-padding length lives in the HDF5, not the CSV.

        With no parts present the column is simply absent rather than an error —
        ``load_labels`` must not require 6.6 GB of waveforms to read a CSV.
        """
        from ecgbench.labels.ikem import load_labels as ikem_labels
        from ecgbench.labels.ikem import read_real_lengths

        root = self._exams(tmp_path, [{"exam_id": 1}])
        assert read_real_lengths(root).empty
        df = ikem_labels(root, self._config(sample_config))
        assert "real_length_samples" not in df.columns
        # The uniform stored length is still known without opening anything.
        assert df.loc[1, "n_samples"] == 4096
        assert df.loc[1, "duration_seconds"] == pytest.approx(8.192)


class TestZZUPediatricLabels:
    """ZZU-pECG: packed code columns that mix codes with prose."""

    #: Two findings with an AHA code, one with none (so AHA is 'N/A').
    ECG_CODE_CSV = (
        "Description,AHA(Category&Code),CHN(Category&Code)\n"
        "Sinus tachycardia,C21,C13\n"
        '"Atrial premature complexes, nonconducted",D31,D22\n'
        "Left ventricular high voltage,N/A,J106\n"
        "Atrial reciprocal beats,N/A,D23\n"
    )
    DISEASE_CODE_CSV = (
        "Disease Type,Disease Category,ICD-10 Code,ICD-10 Description\n"
        "Myocarditis,Acute myocarditis,I40.9,Acute myocarditis\n"
        '"Congenital \nheart disease",Ventricular septal defect,Q21.0,VSD\n'
        "Kawasaki disease,Kawasaki disease,M30.3,Kawasaki\n"
        "Other diseases(OD),Other,See attribute dictionary file,Other\n"
    )
    COLUMNS = (
        "Filename,ECG_ID,Patient_ID,Age,Gender,Acquisition_date,Sampling_point,"
        "Lead,AHA_code,CHN_code,ICD-10 code,pSQI,basSQI,bSQI"
    )

    def _tree(self, tmp_path, rows):
        (tmp_path / "ECGCode.csv").write_text(self.ECG_CODE_CSV, encoding="utf-8")
        (tmp_path / "DiseaseCode.csv").write_text(self.DISEASE_CODE_CSV, encoding="utf-8")
        lines = [self.COLUMNS]
        for row in rows:
            r = {
                "Filename": "P00/P00001/P00001_E01", "ECG_ID": "P00001_E01",
                "Patient_ID": "P00001", "Age": "572d", "Gender": "'Female'",
                "Acquisition_date": "2017-11-22 10:46:08", "Sampling_point": 15000,
                "Lead": 12, "AHA_code": "'C21'", "CHN_code": "'C13'",
                "ICD-10 code": "'Q21.0'",
                "pSQI": "'I':0.288;'II':0.323", "basSQI": "'I':0.98;'II':0.99",
                "bSQI": "'I':1.000;'II':1.000", **row,
            }
            lines.append(",".join(f'"{r[c]}"' for c in self.COLUMNS.split(",")))
        (tmp_path / "AttributesDictionary.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        return replace(
            sample_config, slug="zzu_pecg", record_id_column="ECG_ID",
            patient_id_column="Patient_ID", label_column="aha_codes",
            label_format="comma_separated",
        )

    def test_a_finding_with_no_aha_code_keeps_its_description(
        self, tmp_path, sample_config
    ):
        """ECGCode.csv has no AHA code for 14 of its 105 findings.

        The dataset writes the prose description in the AHA column for exactly
        those, so 6,473 real entries are not codes. Reading the column as a code
        vocabulary invents 15 phantom "codes"; the CHN column names the same
        finding properly, and ``ecg_findings`` carries the description either way.
        """
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels

        root = self._tree(tmp_path, [{
            "AHA_code": "'Left ventricular high voltage';'C21'",
            "CHN_code": "'J106';'C13'",
        }])
        df = zzu_labels(root, self._config(sample_config))
        row = df.loc["P00001_E01"]

        # No AHA code exists, so the description is the only identifier available.
        assert row["aha_codes"] == "Left ventricular high voltage,C21"
        # CHN does have one for it.
        assert row["chn_codes"] == "J106,C13"
        assert row["ecg_findings"] == "Left ventricular high voltage;Sinus tachycardia"
        assert row["n_findings"] == 2

    def test_modifiers_are_stripped_into_a_base_code(self, tmp_path, sample_config):
        """AHA writes L145+Modifier362, CHN writes L121+Depression, and
        J(111+112+113) is a composite. Grouping needs the base code."""
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels

        root = self._tree(tmp_path, [{
            "AHA_code": "'C21+Modifier310'", "CHN_code": "'J(111+112+113)';'C13+Frequent'",
        }])
        row = zzu_labels(root, self._config(sample_config)).loc["P00001_E01"]

        assert row["aha_codes"] == "C21+Modifier310"
        assert row["aha_base_codes"] == "C21"
        assert row["chn_base_codes"] == "J(111+112+113),C13"

    def test_a_comma_in_a_code_would_break_the_label_column_and_is_refused(
        self, tmp_path, sample_config
    ):
        """label_format is comma_separated, so an embedded comma splits one code
        into two bogus ones. Descriptions do contain commas — the fallback path
        is what could introduce one — so this is asserted, not assumed."""
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels

        root = self._tree(tmp_path, [{
            # This description HAS an AHA code (D31), so it normalises safely...
            "AHA_code": "'Atrial premature complexes, nonconducted'",
            "CHN_code": "'D22'",
        }])
        assert zzu_labels(root, self._config(sample_config)).loc[
            "P00001_E01", "aha_codes"
        ] == "D31"

        # ...but one that does not would carry its comma into the list.
        root = self._tree(tmp_path, [{
            "AHA_code": "'Torsades, unspecified'", "CHN_code": "'D22'",
        }])
        with pytest.raises(ValueError, match="comma|','"):
            zzu_labels(root, self._config(sample_config))

    def test_age_is_days_and_the_neonatal_range_survives(self, tmp_path, sample_config):
        """Ages run from 1 day. Rounding to years would collapse every infant."""
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels

        root = self._tree(tmp_path, [{"ECG_ID": "a", "Age": "1d"}])
        row = zzu_labels(root, self._config(sample_config)).loc["a"]
        assert row["age_days"] == 1
        assert row["age_years"] == pytest.approx(1 / 365.25)
        assert row["sex"] == "F"  # "'Female'" -> F, quotes and all

    def test_the_placeholder_disease_row_is_not_treated_as_an_icd_code(
        self, tmp_path, sample_config
    ):
        """DiseaseCode.csv's 20th row is "See attribute dictionary file".

        Matching it as a code would label nothing but would add a phantom group
        to the vocabulary.
        """
        from ecgbench.labels.zzu_pecg import NO_DISEASE
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels

        root = self._tree(tmp_path, [
            {"ECG_ID": "a", "ICD-10 code": "'Q21.0';'J18.9'"},
            {"ECG_ID": "b", "ICD-10 code": "'J18.9'"},
            {"ECG_ID": "c", "ICD-10 code": "'See attribute dictionary file'"},
        ])
        df = zzu_labels(root, self._config(sample_config))

        assert df.loc["a", "disease_groups"] == "Congenital heart disease"
        # The embedded newline in the source's "Congenital \nheart disease" is
        # collapsed, or it would land in a CSV cell verbatim.
        assert "\n" not in df.loc["a", "disease_groups"]
        assert df.loc["b", "disease_groups"] == ""
        assert df.loc["b", "primary_disease_group"] == NO_DISEASE
        assert df.loc["c", "primary_disease_group"] == NO_DISEASE

    def test_sqi_nulls_mark_the_leads_a_reduced_record_lacks(
        self, tmp_path, sample_config
    ):
        """The 9-lead records carry Null for V2/V4/V6, which must not be 0.0."""
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels
        from ecgbench.labels.zzu_pecg import parse_sqi

        parsed = parse_sqi("'I':0.288;'V2':Null;'V5':0.238")
        assert parsed["I"] == pytest.approx(0.288)
        assert np.isnan(parsed["V2"])

        root = self._tree(tmp_path, [{
            "Lead": 9, "pSQI": "'I':0.30;'V2':Null;'V5':0.50",
        }])
        row = zzu_labels(root, self._config(sample_config)).loc["P00001_E01"]
        assert row["n_leads"] == 9
        # Mean over present leads only: (0.30 + 0.50) / 2, not / 3.
        assert row["psqi_mean"] == pytest.approx(0.40)
        assert row["psqi_by_lead"] == "'I':0.30;'V2':Null;'V5':0.50"

    def test_the_signal_path_gets_the_directory_the_csv_omits(
        self, tmp_path, sample_config
    ):
        """Filename is "P00/P00001/P00001_E01"; the files are under Child_ecg/.

        The prefix has to reach the metadata CSV on disk, because
        ``validate_dataset`` rebuilds paths from the raw column and never sees an
        in-memory fix-up. This is the bug Chapman shipped with for months.
        """
        from ecgbench.labels.zzu_pecg import SIGNAL_SUBDIR
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels

        root = self._tree(tmp_path, [{}])
        path = zzu_labels(root, self._config(sample_config)).loc[
            "P00001_E01", "signal_path"
        ]
        assert path == f"{SIGNAL_SUBDIR}/P00/P00001/P00001_E01"
        assert path.startswith("Child_ecg/")

    def test_duration_is_per_record_because_lengths_vary_by_a_factor_of_24(
        self, tmp_path, sample_config
    ):
        """2,500 to 60,000 samples over 67 distinct lengths, so no config value
        could describe it and expected_samples is deliberately empty."""
        from ecgbench.labels.zzu_pecg import load_labels as zzu_labels

        root = self._tree(tmp_path, [
            {"ECG_ID": "short", "Sampling_point": 2500},
            {"ECG_ID": "long", "Sampling_point": 60000},
        ])
        df = zzu_labels(root, self._config(sample_config))
        assert df.loc["short", "duration_seconds"] == pytest.approx(5.0)
        assert df.loc["long", "duration_seconds"] == pytest.approx(120.0)


class TestMedalCareXLLabels:
    """The label is the simulated condition, and the parameters that produced it."""

    def _config(self):
        from ecgbench.config import load_config

        return load_config("medalcare_xl")

    def _root(self, tmp_path, rows=None):
        """Write a stand-in for the generated metadata CSV and its parameter files."""
        rows = rows or [
            ("sinus_train_S65_000001", "S65", "sinus", None, "sinus", "train"),
            ("mi_LAD_1.0_test_S62_000001", "S62", "mi", "LAD_1.0", "mi_LAD_1.0", "test"),
        ]
        records = []
        for rid, model, pathology, subclass, subclass_label, split in rows:
            stem = f"{pathology}/{split}/run_{model}/000001"
            records.append(
                {
                    "record_id": rid,
                    "model_id": model,
                    "pathology": pathology,
                    "mi_subclass": subclass,
                    "mi_occlusion_site": subclass.split("_")[0] if subclass else None,
                    "mi_transmurality": float(subclass.split("_")[1]) if subclass else None,
                    "mi_region": None,
                    "pathology_subclass": subclass_label,
                    "source_split": split,
                    "fold": {"train": 1, "validation": 2, "test": 3}[split],
                    "record_number": "000001",
                    "signal_path": f"WP2_largeDataset_Noise/{stem}_filtered.csv",
                    "signal_path_raw": f"WP2_largeDataset_Noise/{stem}_raw.csv",
                    "signal_path_noise": f"WP2_largeDataset_Noise/{stem}_noise.csv",
                    "atrial_params_path": f"P/{rid}_AtrialParameters.txt",
                    "ventricular_params_path": f"P/{rid}_VentricularParameters.txt",
                }
            )
        pd.DataFrame(records).to_csv(tmp_path / "ecgbench_metadata.csv", index=False)

        # The parameter files, with the ragged key set the real ones have: MI
        # records carry an isch[0].* block that no other pathology does.
        (tmp_path / "P").mkdir(exist_ok=True)
        for record in records:
            (tmp_path / record["atrial_params_path"]).write_text(
                "im.name = Courtemanche\ngeo.atria = cn617_g029\n"
                "geo.torso = torsoID12\ncv_t.BulkTissue = 591mm/s\n"
            )
            ventricular = 'im.name = "MitchellSchaeffer"\nG.torso = 0.22\n'
            if record["pathology"] == "mi":
                ventricular += "isch[0].size = 126.018\nisch[0].tag = 1400.0\n"
            (tmp_path / record["ventricular_params_path"]).write_text(ventricular)
        return tmp_path

    def test_labels_depend_on_the_pipeline_having_run(self, tmp_path):
        """No metadata table ships, so the error must say how to generate one."""
        from ecgbench.labels import LabelSourceMissingError
        from ecgbench.labels.medalcare_xl import load_labels as medal_labels

        with pytest.raises(LabelSourceMissingError) as excinfo:
            medal_labels(tmp_path, self._config())
        message = str(excinfo.value)
        assert "ecgbench_metadata.csv" in message
        assert "ecgbench splits --dataset medalcare_xl" in message

    def test_both_label_layers_are_exposed(self, tmp_path):
        from ecgbench.labels.medalcare_xl import load_labels as medal_labels

        df = medal_labels(self._root(tmp_path), self._config())
        assert df.index.name == "record_id"
        assert list(df["pathology"]) == ["mi", "sinus"]          # sorted by index
        assert list(df["pathology_subclass"]) == ["mi_LAD_1.0", "sinus"]
        assert list(df["pathology_name"]) == [
            "myocardial infarction", "normal sinus rhythm"
        ]
        assert set(df["sampling_rate"]) == {500}

    def test_all_three_signal_variants_are_reachable(self, tmp_path):
        """One record in three renderings — the config wires up only `filtered`."""
        from ecgbench.labels.medalcare_xl import load_labels as medal_labels

        df = medal_labels(self._root(tmp_path), self._config())
        assert df["signal_path"].str.endswith("_filtered.csv").all()
        assert df["signal_path_raw"].str.endswith("_raw.csv").all()
        assert df["signal_path_noise"].str.endswith("_noise.csv").all()

    def test_load_labels_does_not_read_the_parameter_files(self, tmp_path):
        """33,684 file opens on every ECGDataset(labels=True) would not be free."""
        from ecgbench.labels.medalcare_xl import load_labels as medal_labels

        root = self._root(tmp_path)
        # Deleting them must not break the cheap path.
        for path in (root / "P").glob("*.txt"):
            path.unlink()
        df = medal_labels(root, self._config())
        assert len(df) == 2
        assert "atrial.im.name" not in df.columns

    def test_simulation_parameters_are_prefixed_per_provider(self, tmp_path):
        """Both files define im.name and G.torso; an unprefixed concat loses one."""
        from ecgbench.labels.medalcare_xl import load_simulation_parameters

        df = load_simulation_parameters(self._root(tmp_path), self._config())
        assert df.index.name == "record_id"
        assert df.loc["sinus_train_S65_000001", "atrial.im.name"] == "Courtemanche"
        # Quoted in the ventricular files and bare in the atrial ones.
        assert (
            df.loc["sinus_train_S65_000001", "ventricular.im.name"]
            == "MitchellSchaeffer"
        )
        assert not df.columns.duplicated().any()

    def test_absent_parameters_mean_the_pathology_lacks_them(self, tmp_path):
        """MI adds 14 isch[0].* keys; lbbb/rbbb drop stim[*]; lae drops cv_t.*."""
        from ecgbench.labels.medalcare_xl import load_simulation_parameters

        df = load_simulation_parameters(self._root(tmp_path), self._config())
        assert df.loc["mi_LAD_1.0_test_S62_000001", "ventricular.isch[0].tag"] == "1400.0"
        assert pd.isna(df.loc["sinus_train_S65_000001", "ventricular.isch[0].tag"])

    def test_units_travel_with_the_value_rather_than_being_coerced_away(self, tmp_path):
        from ecgbench.labels.medalcare_xl import load_simulation_parameters

        df = load_simulation_parameters(self._root(tmp_path), self._config())
        assert df.loc["sinus_train_S65_000001", "atrial.cv_t.BulkTissue"] == "591mm/s"

    def test_parameters_can_be_restricted_to_one_split(self, tmp_path):
        from ecgbench.labels.medalcare_xl import load_simulation_parameters

        root = self._root(tmp_path)
        df = load_simulation_parameters(
            root, self._config(), record_ids=["sinus_train_S65_000001"]
        )
        assert list(df.index) == ["sinus_train_S65_000001"]
        with pytest.raises(KeyError, match="not in ecgbench_metadata.csv"):
            load_simulation_parameters(root, self._config(), record_ids=["nope"])

    def test_siginfo_is_deliberately_never_read(self):
        """Its rows carry no record number for fam/iab/lae and outnumber the files.

        13 of the 186 run directories have more ``siginfo.csv`` rows than records,
        and for those three pathologies ``info2`` holds a foreign simulation id
        rather than the record number, so any join is a guess about row order. The
        per-record ``*_AtrialParameters.txt`` files carry the same anatomy keyed by
        record number instead. This asserts nobody wires siginfo up later.
        """
        from pathlib import Path

        from ecgbench.labels import medalcare_xl as labels_module
        from ecgbench.splitting.strategies import medalcare_xl as splitter_module

        for module in (labels_module, splitter_module):
            source = Path(module.__file__).read_text()
            code = "\n".join(
                line for line in source.splitlines() if not line.strip().startswith("#")
            )
            # Named only in the docstrings that explain the decision.
            body = code.split('"""')[-1]
            assert "siginfo" not in body, f"{module.__name__} reads siginfo.csv"
