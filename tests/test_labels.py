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
