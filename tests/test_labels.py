"""Tests for per-record label loading."""

from dataclasses import replace

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
