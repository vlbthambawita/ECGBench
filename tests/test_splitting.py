"""Tests for the splitting framework."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import SplitResult
from ecgbench.splitting.engine import split_dataset
from ecgbench.splitting.registry import get_splitter
from ecgbench.splitting.strategies.ecg_arrhythmia import build_metadata


class TestSplitDatasetPredefined:
    """Test splitting with predefined fold assignments."""

    def test_predefined_splits(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        assert isinstance(result, SplitResult)
        assert result.n_folds == 10
        assert result.default_train_folds == [1, 2, 3, 4, 5, 6, 7, 8]
        assert result.default_val_folds == [9]
        assert result.default_test_folds == [10]

    def test_predefined_no_overlap(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        all_ids = set()
        for fold_num, fold_df in result.folds.items():
            fold_ids = set(fold_df["record_id"])
            assert all_ids.isdisjoint(fold_ids), f"Overlap found in fold {fold_num}"
            all_ids.update(fold_ids)

        assert len(all_ids) == len(mock_metadata_with_folds)

    def test_predefined_covers_all(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        total = sum(len(df) for df in result.folds.values())
        assert total == len(mock_metadata_with_folds)


class TestSplitDatasetGrouped:
    """Test splitting with patient-aware grouping (StratifiedGroupKFold)."""

    def test_no_patient_leakage(self, sample_config, mock_metadata_df):
        """ALL records from one patient MUST be in the same fold."""
        labels = mock_metadata_df["label"]
        result = split_dataset(mock_metadata_df, labels, sample_config, n_folds=5)

        patient_fold: dict[str, int] = {}
        for fold_num, fold_df in result.folds.items():
            for pid in fold_df["patient_id"]:
                if pid in patient_fold:
                    assert patient_fold[pid] == fold_num, (
                        f"Patient {pid} appears in folds {patient_fold[pid]} and {fold_num}"
                    )
                else:
                    patient_fold[pid] = fold_num

    def test_deterministic(self, sample_config, mock_metadata_df):
        """Same seed should produce identical results."""
        labels = mock_metadata_df["label"]
        r1 = split_dataset(mock_metadata_df, labels, sample_config, random_state=42)
        r2 = split_dataset(mock_metadata_df, labels, sample_config, random_state=42)

        for fold_num in r1.folds:
            pd.testing.assert_frame_equal(
                r1.folds[fold_num].reset_index(drop=True),
                r2.folds[fold_num].reset_index(drop=True),
            )

    def test_different_seed_different_result(self, sample_config, mock_metadata_df):
        labels = mock_metadata_df["label"]
        r1 = split_dataset(mock_metadata_df, labels, sample_config, random_state=42)
        r2 = split_dataset(mock_metadata_df, labels, sample_config, random_state=99)

        # At least one fold should differ
        any_diff = False
        for fold_num in r1.folds:
            if not r1.folds[fold_num]["record_id"].equals(r2.folds[fold_num]["record_id"]):
                any_diff = True
                break
        assert any_diff


class TestSplitDatasetSimple:
    """Test splitting without patient grouping (StratifiedKFold)."""

    def test_simple_split(self, mock_metadata_df):
        config = DatasetConfig(
            name="test", slug="test", version="1.0", url="http://x",
            metadata_csv="x.csv", record_id_column="record_id",
            label_column="label",
            patient_id_column=None,  # No patient grouping
        )
        labels = mock_metadata_df["label"]
        result = split_dataset(mock_metadata_df, labels, config, n_folds=5)

        assert result.n_folds == 5
        assert result.group_column is None
        total = sum(len(df) for df in result.folds.values())
        assert total == len(mock_metadata_df)


class TestSplitResult:
    def test_train_val_test_properties(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        assert len(result.train) > 0
        assert len(result.val) > 0
        assert len(result.test) > 0
        assert len(result.train) + len(result.val) + len(result.test) == len(
            mock_metadata_with_folds
        )

    def test_get_kfold_split(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        train, val, test = result.get_kfold_split(val_fold=2, test_fold=5)
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(mock_metadata_with_folds)

    def test_get_fold(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        fold_1 = result.get_fold(1)
        assert len(fold_1) > 0

    def test_get_fold_invalid(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        with pytest.raises(ValueError, match="Fold 99 not found"):
            result.get_fold(99)


class TestSplitterRegistry:
    def test_ptbxl_splitter(self):
        splitter = get_splitter("ptbxl")
        assert type(splitter).__name__ == "PTBXLSplitter"

    def test_chapman_splitter(self):
        splitter = get_splitter("chapman_shaoxing")
        assert type(splitter).__name__ == "ChapmanSplitter"

    def test_ecg_arrhythmia_splitter(self):
        splitter = get_splitter("ecg_arrhythmia")
        assert type(splitter).__name__ == "ECGArrhythmiaSplitter"

    def test_mimic_demo_splitter(self):
        splitter = get_splitter("mimic_iv_ecg_demo")
        assert type(splitter).__name__ == "MimicIVECGDemoSplitter"

    def test_ptbdb_splitter(self):
        assert type(get_splitter("ptbdb")).__name__ == "PTBDBSplitter"

    def test_ludb_splitter(self):
        assert type(get_splitter("ludb")).__name__ == "LUDBSplitter"

    def test_unknown_falls_back_to_generic(self):
        splitter = get_splitter("some_unknown_dataset")
        assert type(splitter).__name__ == "GenericSplitter"


class TestECGArrhythmiaSplitter:
    """The ecg-arrhythmia splitter builds its metadata from WFDB headers."""

    def test_build_metadata_scans_headers(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)

        assert len(df) == 15
        assert list(df["record_name"]) == sorted(df["record_name"])
        for col in ("record_name", "signal_path", "age", "sex", "dx",
                    "primary_dx", "primary_dx_acronym", "dx_acronyms"):
            assert col in df.columns

    def test_signal_paths_are_relative_and_point_at_mat_files(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        first = df.loc[df["record_name"] == "JS00001", "signal_path"].item()

        assert first == "WFDBRecords/01/010/JS00001.mat"
        # Resolving against the data root is what validation and ECGDataset do.
        assert not Path(first).is_absolute()

    def test_header_fields_parsed(self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config).set_index(
            "record_name"
        )

        assert df.loc["JS00001", "n_leads"] == 12
        assert df.loc["JS00001", "sampling_rate"] == 500
        assert df.loc["JS00001", "n_samples"] == 5000
        assert str(df.loc["JS00001", "age"]) == "60"
        assert df.loc["JS00013", "sex"] == "Female"

    def test_dx_codes_mapped_to_acronyms(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config).set_index(
            "record_name"
        )

        assert df.loc["JS00001", "primary_dx"] == "426177001"
        assert df.loc["JS00001", "primary_dx_acronym"] == "SB"
        assert df.loc["JS00001", "dx_acronyms"] == "SB,TWC"
        # Codes absent from ConditionNames_SNOMED-CT.csv stay as raw codes.
        assert df.loc["JS00014", "primary_dx_acronym"] == "55827005"

    def test_malformed_record_line_is_tolerated(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        """A record ECGBench cannot parse positionally is still listed.

        The validation engine is what flags it, via corrupt_header.
        """
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config).set_index(
            "record_name"
        )

        assert "JS00015" in df.index
        assert pd.isna(df.loc["JS00015", "n_samples"])
        assert df.loc["JS00015", "primary_dx_acronym"] == "SB"

    def test_load_metadata_caches_csv(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        """The cache must land on disk — validate_dataset re-reads it from there."""
        splitter = get_splitter("ecg_arrhythmia")
        csv_path = tmp_ecg_arrhythmia_data / ecg_arrhythmia_config.metadata_csv
        assert not csv_path.exists()

        first = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        assert csv_path.exists()

        second = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        pd.testing.assert_frame_equal(
            first[["record_name", "signal_path", "primary_dx_acronym"]],
            second[["record_name", "signal_path", "primary_dx_acronym"]],
        )

    def test_stratification_pools_rare_classes(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        splitter = get_splitter("ecg_arrhythmia")
        df = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        labels = splitter.get_stratification_labels(df, ecg_arrhythmia_config)

        assert labels.name == "primary_diagnosis"
        assert len(labels) == len(df)
        counts = labels.value_counts()
        # 13 SB records survive as their own class; AFIB and the unmapped code
        # have one record each and are pooled.
        assert counts["SB"] == 13
        assert counts["OTHER"] == 2
        assert "AFIB" not in counts

    def test_split_is_ungrouped(self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data):
        """No patient_id_column, so StratifiedKFold — not the grouped variant."""
        splitter = get_splitter("ecg_arrhythmia")
        df = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        labels = splitter.get_stratification_labels(df, ecg_arrhythmia_config)

        result = split_dataset(df, labels, ecg_arrhythmia_config, n_folds=3)

        assert result.group_column is None
        assert result.split_metadata["method"] == "StratifiedKFold"
        assert result.n_folds == 3
        assert sum(len(f) for f in result.folds.values()) == len(df)


class TestMimicIVECGDemoSplitter:
    """The MIMIC demo has no labels, so grouping is the only real guarantee."""

    def test_load_metadata_reads_shipped_csv(self, mimic_demo_config, tmp_mimic_demo_data):
        splitter = get_splitter("mimic_iv_ecg_demo")
        df = splitter.load_metadata(tmp_mimic_demo_data, mimic_demo_config)

        assert len(df) == 20
        assert df["subject_id"].nunique() == 5
        # Paths are used as-is: relative to the data root and extension-free.
        first = df.loc[df["study_id"] == 100000320, "path"].item()
        assert first == "files/p10000032/s100000320/100000320"
        assert Path(first).suffix == ""

    def test_label_column_is_materialised(self, mimic_demo_config, tmp_mimic_demo_data):
        """The config declares label_column, so the column must actually exist."""
        splitter = get_splitter("mimic_iv_ecg_demo")
        df = splitter.load_metadata(tmp_mimic_demo_data, mimic_demo_config)

        assert mimic_demo_config.label_column in df.columns
        assert set(df[mimic_demo_config.label_column]) == {"UNLABELLED"}

    def test_missing_columns_raise(self, mimic_demo_config, tmp_mimic_demo_data):
        pd.DataFrame({"study_id": [1], "path": ["files/x/y/1"]}).to_csv(
            tmp_mimic_demo_data / "record_list.csv", index=False
        )
        splitter = get_splitter("mimic_iv_ecg_demo")

        with pytest.raises(ValueError, match="missing expected columns"):
            splitter.load_metadata(tmp_mimic_demo_data, mimic_demo_config)

    def test_stratification_is_single_class(self, mimic_demo_config, tmp_mimic_demo_data):
        splitter = get_splitter("mimic_iv_ecg_demo")
        df = splitter.load_metadata(tmp_mimic_demo_data, mimic_demo_config)
        labels = splitter.get_stratification_labels(df, mimic_demo_config)

        assert labels.nunique() == 1
        assert len(labels) == len(df)
        assert labels.index.equals(df.index)

    def test_subjects_never_span_folds(self, mimic_demo_config, tmp_mimic_demo_data):
        """The one guarantee that matters: 20 records from only 5 subjects."""
        splitter = get_splitter("mimic_iv_ecg_demo")
        df = splitter.load_metadata(tmp_mimic_demo_data, mimic_demo_config)
        labels = splitter.get_stratification_labels(df, mimic_demo_config)

        result = split_dataset(df, labels, mimic_demo_config, n_folds=3)

        assert result.group_column == "subject_id"
        assert result.split_metadata["method"] == "StratifiedGroupKFold"
        subject_fold: dict[int, int] = {}
        for fold_num, fold_df in result.folds.items():
            for subject_id in fold_df["subject_id"]:
                assert subject_fold.setdefault(subject_id, fold_num) == fold_num, (
                    f"Subject {subject_id} appears in more than one fold"
                )
        assert sum(len(f) for f in result.folds.values()) == len(df)


class TestChapmanSplitter:
    """Chapman normalises xlsx/csv metadata and builds ECGData/<name>.csv paths."""

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config,
            slug="chapman_shaoxing",
            metadata_csv="ecgbench_metadata.csv",
            record_id_column="FileName",
            patient_id_column=None,
            signal_path_columns={500: "signal_path"},
            label_column="Rhythm",
            signal_format="csv",
            signal_unit_scale=0.001,
        )

    def _source(self, tmp_path):
        pd.DataFrame({
            "FileName": ["MUSE_C", "MUSE_A", "MUSE_B"],
            "Rhythm": ["AFIB", "SB", "SB"],
            "Beat": ["RBBB TWC", "NONE", "TWC"],
            "PatientAge": [85, 59, 20],
            "Gender": ["MALE", "FEMALE", "FEMALE"],
        }).to_csv(tmp_path / "Diagnostics.csv", index=False)
        (tmp_path / "ECGData").mkdir()
        return tmp_path

    def test_signal_paths_get_dir_and_extension(self, sample_config, tmp_path):
        """FileName is a bare stem: both ECGData/ and .csv have to be added."""
        from ecgbench.splitting.strategies.chapman import build_metadata

        config = self._config(sample_config)
        df = build_metadata(self._source(tmp_path), config)

        assert list(df["FileName"]) == ["MUSE_A", "MUSE_B", "MUSE_C"]  # sorted
        assert df.loc[0, "signal_path"] == "ECGData/MUSE_A.csv"

    def test_metadata_is_cached_to_disk(self, sample_config, tmp_path):
        """validate_dataset re-reads metadata_csv, so it must land on disk."""
        splitter = get_splitter("chapman_shaoxing")
        config = self._config(sample_config)
        source = self._source(tmp_path)
        generated = source / config.metadata_csv
        assert not generated.exists()

        first = splitter.load_metadata(source, config)
        assert generated.exists()
        # The cached file must carry the fixed-up path, not the bare stem — that
        # asymmetry is what used to make every record fail corrupt_header.
        assert "ECGData/MUSE_A.csv" in generated.read_text()

        second = splitter.load_metadata(source, config)
        pd.testing.assert_frame_equal(first, second)

    def test_missing_signal_dir_raises(self, sample_config, tmp_path):
        from ecgbench.splitting.strategies.chapman import build_metadata

        pd.DataFrame({"FileName": ["x"], "Rhythm": ["SB"]}).to_csv(
            tmp_path / "Diagnostics.csv", index=False
        )
        with pytest.raises(FileNotFoundError, match="signal directory"):
            build_metadata(tmp_path, self._config(sample_config))

    def test_missing_source_metadata_raises(self, sample_config, tmp_path):
        from ecgbench.splitting.strategies.chapman import build_metadata

        with pytest.raises(FileNotFoundError, match="Diagnostics"):
            build_metadata(tmp_path, self._config(sample_config))

    def test_stratifies_on_rhythm(self, sample_config, tmp_path):
        splitter = get_splitter("chapman_shaoxing")
        config = self._config(sample_config)
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "rhythm"
        assert labels.value_counts().to_dict() == {"SB": 2, "AFIB": 1}


class TestPTBDBSplitter:
    """PTBDB derives everything from the .hea comment blocks."""

    HEADER = (
        "s0001_re 15 1000 38400\r\n"
        "s0001_re.dat 16 2000 16 0 -489 -8337 0 i\r\n"
        "# age: 81\r\n"
        "# sex: female\r\n"
        "# Diagnose:\r\n"
        "# Reason for admission: Myocardial infarction\r\n"
        "# Additional diagnoses: n/a\r\n"
        "# Catheterization date: 01/10/1990\r\n"
        "# Therapy:\r\n"
        "# Catheterization date: 02/10/1990\r\n"
    )

    def _tree(self, tmp_path, diagnoses):
        """One patient dir per diagnosis, some with two records."""
        for i, diagnosis in enumerate(diagnoses, start=1):
            d = tmp_path / f"patient{i:03d}"
            d.mkdir()
            for j in range(1, 3 if i % 2 == 0 else 2):  # every other patient: 2 records
                header = self.HEADER.replace("Myocardial infarction", diagnosis)
                (d / f"s{i:03d}{j}_re.hea").write_text(header, encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ptbdb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"}, default_sampling_rate=1000,
            label_column="diagnosis", leads=15,
        )

    def test_parses_crlf_comment_block(self, tmp_path):
        from ecgbench.labels.ptbdb import parse_header_comments

        (tmp_path / "x.hea").write_text(self.HEADER, encoding="utf-8")
        fields = parse_header_comments(tmp_path / "x.hea")

        assert fields["age"] == "81"
        assert fields["sex"] == "female"
        assert fields["Reason for admission"] == "Myocardial infarction"
        # 'n/a' is the dataset's absence marker, normalised to empty.
        assert fields["Additional diagnoses"] == ""
        # The repeated key is kept, not silently overwritten.
        assert fields["Catheterization date"] == "01/10/1990"
        assert fields["Catheterization date (2)"] == "02/10/1990"

    def test_builds_patient_id_and_signal_path(self, sample_config, tmp_path):
        splitter = get_splitter("ptbdb")
        config = self._config(sample_config)
        df = splitter.load_metadata(self._tree(tmp_path, ["Healthy control"] * 3), config)

        assert set(df["patient_id"]) == {"patient001", "patient002", "patient003"}
        assert df.loc[0, "signal_path"].startswith("patient001/")
        assert (tmp_path / config.metadata_csv).exists()

    def test_missing_records_raise(self, sample_config, tmp_path):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="patient..../..hea|headers"):
            get_splitter("ptbdb").load_metadata(tmp_path, self._config(sample_config))

    def test_rare_diagnoses_pool_and_missing_becomes_unknown(self, sample_config, tmp_path):
        from ecgbench.labels.ptbdb import OTHER, UNKNOWN

        splitter = get_splitter("ptbdb")
        config = self._config(sample_config)
        # 12 controls survive; one Myocarditis and one undiagnosed do not.
        tree = self._tree(tmp_path, ["Healthy control"] * 8 + ["Myocarditis", "n/a"])
        labels = splitter.get_stratification_labels(
            splitter.load_metadata(tree, config), config
        )
        counts = labels.value_counts().to_dict()

        assert counts["Healthy control"] >= 10
        assert OTHER in counts          # Myocarditis pooled
        assert UNKNOWN not in counts    # only one, so it pooled into OTHER too


class TestLUDBSplitter:
    """LUDB's CSV is newline-polluted and multi-label; the loader normalises it."""

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ludb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="ID", patient_id_column=None,
            signal_path_columns={500: "signal_path"}, label_column="primary_rhythm",
        )

    def _source(self, tmp_path, n=12):
        rows = []
        for i in range(1, n + 1):
            rows.append({
                "ID": i,
                "Sex": "F\n" if i % 2 else "M\n",
                "Age": ">89\n" if i == 1 else f"{40 + i}\n",
                "Rhythms": "Sinus rhythm\n" if i > 2 else "Atrial fibrillation\n",
                "Electric axis of the heart": "Electric axis of the heart: normal\n",
                "Conduction abnormalities": None,
                "Extrasystolies": None,
                "Hypertrophies": "Left atrial hypertrophy\nLeft ventricular hypertrophy\n",
                "Cardiac pacing": None,
                "Ischemia": None,
                "Non-specific repolarization abnormalities": None,
                "Other states": None,
            })
        pd.DataFrame(rows).to_csv(tmp_path / "ludb.csv", index=False)
        (tmp_path / "data").mkdir()
        return tmp_path

    def test_strips_trailing_newlines(self, sample_config, tmp_path):
        from ecgbench.labels.ludb import load_labels

        df = load_labels(self._source(tmp_path), self._config(sample_config))

        assert df.loc[2, "sex"] == "M"           # was 'M\n'
        assert df.loc[3, "electric_axis"] == "normal"  # prefix and dot stripped

    def test_multi_label_cells_become_lists(self, sample_config, tmp_path):
        from ecgbench.labels.ludb import load_labels

        df = load_labels(self._source(tmp_path), self._config(sample_config))

        assert df.loc[1, "hypertrophies"] == [
            "Left atrial hypertrophy", "Left ventricular hypertrophy",
        ]
        assert df.loc[1, "conduction_abnormalities"] == []

    def test_non_numeric_age_is_kept_raw(self, sample_config, tmp_path):
        from ecgbench.labels.ludb import load_labels

        df = load_labels(self._source(tmp_path), self._config(sample_config))

        assert df.loc[1, "age_raw"] == ">89"
        assert pd.isna(df.loc[1, "age"])
        assert df.loc[2, "age"] == 42

    def test_signal_paths_and_list_flattening(self, sample_config, tmp_path):
        splitter = get_splitter("ludb")
        config = self._config(sample_config)
        df = splitter.load_metadata(self._source(tmp_path), config)

        assert df.loc[0, "signal_path"] == "data/1"
        # Lists would round-trip through CSV as their repr, so they are joined.
        assert df.loc[0, "hypertrophies"] == (
            "Left atrial hypertrophy;Left ventricular hypertrophy"
        )

    def test_missing_data_dir_raises(self, sample_config, tmp_path):
        pd.DataFrame({"ID": [1]}).to_csv(tmp_path / "ludb.csv", index=False)
        with pytest.raises(FileNotFoundError, match="signal directory"):
            get_splitter("ludb").load_metadata(tmp_path, self._config(sample_config))


class TestChallenge2021Splitter:
    """Challenge 2021 pools eight cohorts and has no clinically primary diagnosis."""

    def _header(self, name, dx, fs=500, nsamp=5000, age="54", sex="Female"):
        return (
            f"{name} 12 {fs} {nsamp}\n"
            + "".join(
                f"{name}.mat 16x1+24 1000.0(0)/mV 16 0 0 0 0 {lead}\n"
                for lead in ("I", "II", "III", "aVR", "aVL", "aVF",
                             "V1", "V2", "V3", "V4", "V5", "V6")
            )
            + f"# Age: {age}\n# Sex: {sex}\n# Dx: {dx}\n"
            "# Rx: Unknown\n# Hx: Unknown\n# Sx: Unknown\n"
        )

    def _tree(self, tmp_path, records):
        """records: {cohort: [(name, dx), ...]} under training/<cohort>/g1/."""
        for cohort, items in records.items():
            d = tmp_path / "training" / cohort / "g1"
            d.mkdir(parents=True, exist_ok=True)
            for name, dx in items:
                (d / f"{name}.hea").write_text(self._header(name, dx), encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="challenge2021", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={500: "signal_path"}, default_sampling_rate=500,
            label_column="dx", leads=12,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.challenge2021 import Challenge2021Splitter

        assert isinstance(get_splitter("challenge2021"), Challenge2021Splitter)

    def test_builds_source_and_signal_path_and_caches_csv(self, sample_config, tmp_path):
        config = self._config(sample_config)
        tree = self._tree(tmp_path, {
            "ningbo": [("JS10647", "426783006")],
            "ptb-xl": [("HR00001", "164889003")],
        })
        df = get_splitter("challenge2021").load_metadata(tree, config)

        assert set(df["source"]) == {"ningbo", "ptb-xl"}
        # Paths are relative to data_path and point at the .mat, not the .hea.
        paths = dict(zip(df["record_name"], df["signal_path"]))
        assert paths["JS10647"] == "training/ningbo/g1/JS10647.mat"
        assert paths["HR00001"] == "training/ptb-xl/g1/HR00001.mat"
        # Written to disk because validate_dataset re-reads it rather than
        # reusing this frame.
        assert (tree / config.metadata_csv).exists()

    def test_multi_label_codes_map_to_abbreviations(self, sample_config, tmp_path):
        config = self._config(sample_config)
        # AF + RBBB + TAb, in the order a real Chapman header lists them.
        tree = self._tree(tmp_path, {
            "chapman_shaoxing": [("JS00001", "164889003,59118001,164934002")],
        })
        row = get_splitter("challenge2021").load_metadata(tree, config).iloc[0]

        assert row["n_dx"] == 3
        assert row["dx_abbreviations"] == "AF,RBBB,TAb"
        assert row["scored_dx"] == "AF,RBBB,TAb"  # all three are scored classes
        assert "atrial fibrillation" in row["dx_names"]

    def test_stratification_takes_the_rarest_code_not_the_first(
        self, sample_config, tmp_path
    ):
        """The reduction must not depend on #Dx ordering, which is meaningless here."""
        config = self._config(sample_config)
        # NSR is common in this tiny corpus; AF appears once. A record carrying
        # both must stratify on AF regardless of which is listed first.
        tree = self._tree(tmp_path, {"ningbo": (
            [(f"JS1{i:04d}", "426783006") for i in range(12)]
            + [("JS19999", "426783006,164889003")]
        )})
        splitter = get_splitter("challenge2021")
        df = splitter.load_metadata(tree, config)

        strat = dict(zip(df["record_name"], df["stratify_dx_abbreviation"]))
        assert strat["JS19999"] == "AF"     # rarest, though NSR is listed first
        assert strat["JS10000"] == "NSR"

    def test_rare_classes_pool_into_other(self, sample_config, tmp_path):
        from ecgbench.splitting.strategies.challenge2021 import OTHER

        config = self._config(sample_config)
        tree = self._tree(tmp_path, {"ningbo": (
            [(f"JS1{i:04d}", "426783006") for i in range(12)]
            + [("JS19999", "164889003")]
        )})
        splitter = get_splitter("challenge2021")
        labels = splitter.get_stratification_labels(
            splitter.load_metadata(tree, config), config
        )
        counts = labels.value_counts().to_dict()

        assert counts["NSR"] == 12
        assert counts[OTHER] == 1   # the single AF record
        assert "AF" not in counts

    def test_stratification_needs_load_metadata_first(self, sample_config, tmp_path):
        import pandas as pd

        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("challenge2021").get_stratification_labels(
                pd.DataFrame({"record_name": ["A0001"]}), self._config(sample_config)
            )

    def test_missing_training_tree_raises(self, sample_config, tmp_path):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="training"):
            get_splitter("challenge2021").load_metadata(
                tmp_path, self._config(sample_config)
            )


class TestChallenge2020Splitter:
    """Challenge 2020 is Challenge 2021's six-cohort subset, with a label defect."""

    def _header(self, name, dx, fs=500, nsamp=5000, age="54", sex="Female"):
        return (
            f"{name} 12 {fs} {nsamp}\n"
            + "".join(
                f"{name}.mat 16x1+24 1000.0(0)/mV 16 0 0 0 0 {lead}\n"
                for lead in ("I", "II", "III", "aVR", "aVL", "aVF",
                             "V1", "V2", "V3", "V4", "V5", "V6")
            )
            + f"# Age: {age}\n# Sex: {sex}\n# Dx: {dx}\n"
            "# Rx: Unknown\n# Hx: Unknown\n# Sx: Unknown\n"
        )

    def _tree(self, tmp_path, records):
        """records: {cohort: [(name, dx), ...]} under training/<cohort>/g1/."""
        for cohort, items in records.items():
            d = tmp_path / "training" / cohort / "g1"
            d.mkdir(parents=True, exist_ok=True)
            for name, dx in items:
                (d / f"{name}.hea").write_text(self._header(name, dx), encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="challenge2020", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={500: "signal_path"}, default_sampling_rate=500,
            label_column="dx", leads=12,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.challenge2020 import Challenge2020Splitter

        assert isinstance(get_splitter("challenge2020"), Challenge2020Splitter)

    def test_builds_source_and_signal_path_and_caches_csv(self, sample_config, tmp_path):
        config = self._config(sample_config)
        tree = self._tree(tmp_path, {
            "ptb-xl": [("HR00001", "164889003")],
            "georgia": [("E00001", "426783006")],
        })
        df = get_splitter("challenge2020").load_metadata(tree, config)

        assert set(df["source"]) == {"ptb-xl", "georgia"}
        # Paths are relative to data_path and point at the .mat, not the .hea.
        paths = dict(zip(df["record_name"], df["signal_path"]))
        assert paths["HR00001"] == "training/ptb-xl/g1/HR00001.mat"
        assert paths["E00001"] == "training/georgia/g1/E00001.mat"
        # Written to disk because validate_dataset re-reads it rather than
        # reusing this frame.
        assert (tree / config.metadata_csv).exists()

    def test_repeated_dx_codes_do_not_inflate_the_metadata(self, sample_config, tmp_path):
        """628 Georgia records ship a repeated code; the splitter must see one.

        Left in, the duplicates change n_dx, the multi-hot target width and the
        code frequencies the stratification reduction is computed from.
        """
        config = self._config(sample_config)
        tree = self._tree(tmp_path, {
            "georgia": [("E00015", "251146004,284470004,284470004")],
        })
        row = get_splitter("challenge2020").load_metadata(tree, config).iloc[0]

        assert row["dx"] == "251146004,284470004"
        assert row["n_dx"] == 2
        assert row["dx_abbreviations"] == "LQRSV,PAC"

    def test_sex_is_normalised_across_cohorts(self, sample_config, tmp_path):
        """St Petersburg spells sex M/F; everyone else spells it out."""
        config = self._config(sample_config)
        d = tmp_path / "training" / "st_petersburg_incart" / "g1"
        d.mkdir(parents=True)
        (d / "I0001.hea").write_text(
            self._header("I0001", "426783006", fs=257, nsamp=462600, sex="F"),
            encoding="utf-8",
        )
        d2 = tmp_path / "training" / "ptb-xl" / "g1"
        d2.mkdir(parents=True)
        (d2 / "HR00001.hea").write_text(
            self._header("HR00001", "426783006", sex="Female"), encoding="utf-8"
        )
        df = get_splitter("challenge2020").load_metadata(tmp_path, config)

        assert set(df["sex"]) == {"Female"}

    def test_stratification_takes_the_rarest_code_not_the_first(
        self, sample_config, tmp_path
    ):
        """The reduction must not depend on #Dx ordering, which is meaningless here."""
        config = self._config(sample_config)
        tree = self._tree(tmp_path, {"ptb-xl": (
            [(f"HR{i:05d}", "426783006") for i in range(12)]
            + [("HR99999", "426783006,164889003")]
        )})
        df = get_splitter("challenge2020").load_metadata(tree, config)

        strat = dict(zip(df["record_name"], df["stratify_dx_abbreviation"]))
        assert strat["HR99999"] == "AF"     # rarest, though NSR is listed first
        assert strat["HR00000"] == "NSR"

    def test_rare_classes_pool_into_other(self, sample_config, tmp_path):
        from ecgbench.splitting.strategies.challenge2020 import OTHER

        config = self._config(sample_config)
        tree = self._tree(tmp_path, {"ptb-xl": (
            [(f"HR{i:05d}", "426783006") for i in range(12)]
            + [("HR99999", "164889003")]
        )})
        splitter = get_splitter("challenge2020")
        labels = splitter.get_stratification_labels(
            splitter.load_metadata(tree, config), config
        )
        counts = labels.value_counts().to_dict()

        assert counts["NSR"] == 12
        assert counts[OTHER] == 1   # the single AF record
        assert "AF" not in counts

    def test_stratification_needs_load_metadata_first(self, sample_config, tmp_path):
        import pandas as pd

        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("challenge2020").get_stratification_labels(
                pd.DataFrame({"record_name": ["A0001"]}), self._config(sample_config)
            )

    def test_missing_training_tree_raises(self, sample_config, tmp_path):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="training"):
            get_splitter("challenge2020").load_metadata(
                tmp_path, self._config(sample_config)
            )


class TestINCARTDBSplitter:
    """INCART's binding constraint is patient grouping: 75 records, 32 patients."""

    def _header(self, rec, patient, dx, features, gain=306):
        dx_part = f" <diagnoses> {dx}" if dx else " "
        return (
            f"{rec} 12 257 462600\n"
            + "".join(
                f"{rec}.dat 16 {gain} 16 0 0 0 0 {lead}\n"
                for lead in ("I", "II", "III", "AVR", "AVL", "AVF",
                             "V1", "V2", "V3", "V4", "V5", "V6")
            )
            + f"#<age>: 65 <sex>: F{dx_part}\n"
            f"# patient {patient}\n"
            f"# {features}\n"
        )

    def _tree(self, tmp_path, records):
        """records: [(rec, patient, dx)] -> flat I??.hea files, plus .atr each."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, patient, dx in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(rec, patient, dx, "PVCs, noise"), encoding="utf-8"
            )
            wfdb.wrann(
                rec, "atr",
                sample=np.array([100, 200, 300]),
                symbol=["N", "N", "V"],
                write_dir=str(tmp_path),
            )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="incartdb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="patient_id",
            signal_path_columns={257: "signal_path"}, default_sampling_rate=257,
            label_column="diagnosis", leads=12,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.incartdb import INCARTDBSplitter

        assert isinstance(get_splitter("incartdb"), INCARTDBSplitter)

    def test_builds_patient_id_signal_path_and_beat_counts(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("I01", 1, "Acute MI"), ("I02", 1, "Acute MI")])
        df = get_splitter("incartdb").load_metadata(tree, config)

        # Both records belong to one patient — that is what grouping depends on.
        assert set(df["patient_id"]) == {"patient01"}
        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["I01", "I02"]
        assert df["n_beats"].tolist() == [3, 3]
        assert df["beat_V"].tolist() == [1, 1]
        assert df["pvc_fraction"].round(4).tolist() == [0.3333, 0.3333]
        # Written to disk because validate_dataset re-reads it.
        assert (tree / config.metadata_csv).exists()

    def test_records_without_a_diagnosis_are_labelled_unknown(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        from ecgbench.labels.incartdb import UNKNOWN

        config = self._config(sample_config)
        # 10 patients with no diagnosis, so UNKNOWN clears MIN_CLASS_PATIENTS and
        # survives pooling — as it does in the real dataset, where 14 patients
        # have none. With fewer patients it would pool into OTHER like any other
        # small class; UNKNOWN is not privileged.
        records = [(f"I{i:02d}", i, "") for i in range(1, 11)]
        df = get_splitter("incartdb").load_metadata(self._tree(tmp_path, records), config)

        assert set(df["diagnosis"]) == {""}
        labels = get_splitter("incartdb").get_stratification_labels(df, config)
        assert set(labels) == {UNKNOWN}

    def test_no_patient_spans_a_fold(self, sample_config, tmp_path):
        """The property that makes this dataset usable at all."""
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        # 12 patients, 2-3 records each, two diagnosis classes.
        records = []
        rec = 1
        for patient in range(1, 13):
            dx = "Acute MI" if patient % 2 else ""
            for _ in range(2 if patient % 3 else 3):
                records.append((f"I{rec:02d}", patient, dx))
                rec += 1
        splitter = get_splitter("incartdb")
        df = splitter.load_metadata(self._tree(tmp_path, records), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=4)

        assigned = pd.concat(
            [f.assign(fold=n) for n, f in result.folds.items()], ignore_index=True
        )
        spanning = assigned.groupby("patient_id")["fold"].nunique()
        assert (spanning == 1).all(), f"patients split across folds: {spanning[spanning > 1]}"
        assert result.group_column == "patient_id"

    def test_stratification_needs_load_metadata_first(self, sample_config):
        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("incartdb").get_stratification_labels(
                pd.DataFrame({"record_name": ["I01"]}), self._config(sample_config)
            )

    def test_missing_records_raise(self, sample_config, tmp_path):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="I..\\.hea|headers"):
            get_splitter("incartdb").load_metadata(tmp_path, self._config(sample_config))


class TestMimicIVECGSplitter:
    """The full MIMIC-IV-ECG release: report-stratified, subject-grouped."""

    def _tree(self, tmp_path, records, measurements=True):
        """records: [(study_id, subject_id, report_0)] -> record_list.csv (+ measurements)."""
        pd.DataFrame({
            "subject_id": [r[1] for r in records],
            "study_id": [r[0] for r in records],
            "file_name": [str(r[0]) for r in records],
            "ecg_time": ["2180-07-23 08:44:00"] * len(records),
            "path": [f"files/p1000/p{r[1]}/s{r[0]}/{r[0]}" for r in records],
        }).to_csv(tmp_path / "record_list.csv", index=False)

        if measurements:
            mm = pd.DataFrame({
                "subject_id": [r[1] for r in records],
                "study_id": [r[0] for r in records],
                "cart_id": [1] * len(records),
                "ecg_time": ["2180-07-23 08:44:00"] * len(records),
                "report_0": [r[2] for r in records],
                "rr_interval": [800] * len(records),
                "p_onset": [40] * len(records),
                "p_end": [150] * len(records),
                "qrs_onset": [200] * len(records),
                "qrs_end": [290] * len(records),
                "t_end": [610] * len(records),
                "p_axis": [50] * len(records),
                "qrs_axis": [13] * len(records),
                "t_axis": [42] * len(records),
                "bandwidth": ["0.5-40"] * len(records),
                "filtering": ["60Hz"] * len(records),
            })
            for i in range(1, 18):
                mm[f"report_{i}"] = None
            mm.to_csv(tmp_path / "machine_measurements.csv", index=False)
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        from ecgbench.config import LabelConfig

        return replace(
            sample_config, slug="mimic_iv_ecg", metadata_csv="record_list.csv",
            record_id_column="study_id", patient_id_column="subject_id",
            signal_path_columns={500: "path"}, default_sampling_rate=500,
            label_column="stratify_class", leads=12,
            labels=LabelConfig(source_csv="machine_measurements.csv",
                               join_column="study_id"),
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.mimic_iv_ecg import MimicIVECGSplitter

        assert isinstance(get_splitter("mimic_iv_ecg"), MimicIVECGSplitter)

    def test_reads_record_list_and_joins_the_report_class(self, sample_config, tmp_path):
        """The class must come from the label loader, normalised and pooled by it."""
        from ecgbench.labels.mimic_iv_ecg import MIN_CLASS_SIZE, OTHER

        config = self._config(sample_config)
        # Two classes big enough to survive pooling, plus one genuinely rare
        # report. 'Sinus rhythm.' and '  Sinus   Rhythm ' must normalise together,
        # otherwise neither reaches MIN_CLASS_SIZE.
        records, study = [], 1
        for i in range(MIN_CLASS_SIZE):
            records.append((study, 100 + i, "Sinus rhythm." if i % 2 else "  Sinus   Rhythm "))
            study += 1
        for i in range(MIN_CLASS_SIZE):
            records.append((study, 9000 + i, "Atrial fibrillation"))
            study += 1
        records.append((study, 99999, "Some very rare finding"))
        rare_study = study

        df = get_splitter("mimic_iv_ecg").load_metadata(self._tree(tmp_path, records), config)
        classes = df.set_index("study_id")["stratify_class"]

        assert len(df) == 2 * MIN_CLASS_SIZE + 1
        # Paths are used verbatim — record_list.csv needs no fix-up.
        assert df.loc[0, "path"] == "files/p1000/p100/s1/1"
        assert classes.loc[1] == "sinus rhythm"
        assert classes.loc[MIN_CLASS_SIZE + 1] == "atrial fibrillation"
        assert (classes == "sinus rhythm").sum() == MIN_CLASS_SIZE
        # The singleton pools rather than becoming its own fold-breaking class.
        assert classes.loc[rare_study] == OTHER

    def test_absent_measurements_file_degrades_to_grouping_only(
        self, sample_config, tmp_path, caplog
    ):
        """A partial copy must still produce a reproducible patient-grouped split."""
        from ecgbench.labels.mimic_iv_ecg import UNKNOWN

        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(1, 100, "Sinus rhythm")], measurements=False)

        with caplog.at_level("WARNING"):
            df = get_splitter("mimic_iv_ecg").load_metadata(tree, config)

        assert set(df["stratify_class"]) == {UNKNOWN}
        assert "purely patient-grouped" in caplog.text

    def test_partial_measurements_coverage_warns(self, sample_config, tmp_path, caplog):
        """A filtered local machine_measurements.csv is the trap this catches."""
        from ecgbench.labels.mimic_iv_ecg import UNKNOWN

        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(1, 100, "Sinus rhythm"), (2, 100, "Sinus rhythm")])
        # Drop study 2 from the measurements file only, as a filtered copy would.
        mm = pd.read_csv(tree / "machine_measurements.csv")
        mm[mm.study_id != 2].to_csv(tree / "machine_measurements.csv", index=False)

        with caplog.at_level("WARNING"):
            df = get_splitter("mimic_iv_ecg").load_metadata(tree, config)

        assert df.set_index("study_id").loc[2, "stratify_class"] == UNKNOWN
        assert "SHA256SUMS" in caplog.text

    def test_missing_record_list_names_the_expected_layout(self, sample_config, tmp_path):
        with pytest.raises(FileNotFoundError, match="record_list.csv"):
            get_splitter("mimic_iv_ecg").load_metadata(tmp_path, self._config(sample_config))

    def test_missing_columns_are_reported(self, sample_config, tmp_path):
        pd.DataFrame({"study_id": [1], "nope": ["x"]}).to_csv(
            tmp_path / "record_list.csv", index=False
        )
        with pytest.raises(ValueError, match="missing expected columns"):
            get_splitter("mimic_iv_ecg").load_metadata(tmp_path, self._config(sample_config))

    def test_no_subject_spans_a_fold(self, sample_config, tmp_path):
        """64.5% of real subjects have several studies, so this is the binding property."""
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        records, study = [], 1
        for subject in range(100, 130):
            report = "Sinus rhythm" if subject % 2 else "Atrial fibrillation"
            for _ in range(2 if subject % 3 else 4):
                records.append((study, subject, report))
                study += 1
        splitter = get_splitter("mimic_iv_ecg")
        df = splitter.load_metadata(self._tree(tmp_path, records), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assigned = pd.concat(
            [f.assign(fold=n) for n, f in result.folds.items()], ignore_index=True
        )
        spanning = assigned.groupby("subject_id")["fold"].nunique()
        assert (spanning == 1).all(), f"subjects split across folds: {spanning[spanning > 1]}"
        assert result.group_column == "subject_id"

    def test_stratification_needs_load_metadata_first(self, sample_config):
        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("mimic_iv_ecg").get_stratification_labels(
                pd.DataFrame({"study_id": [1]}), self._config(sample_config)
            )


class TestBrugadaHUCASplitter:
    """Brugada-HUCA needs a splitter only because metadata.csv has no path column."""

    def _tree(self, tmp_path, rows, stray_ds_store=True, duplicate_in_records=True):
        """rows: [(patient_id, brugada, basal_pattern, sudden_death)]."""
        pd.DataFrame(
            rows, columns=["patient_id", "brugada", "basal_pattern", "sudden_death"]
        ).to_csv(tmp_path / "metadata.csv", index=False)

        files = tmp_path / "files"
        files.mkdir()
        for pid, *_ in rows:
            d = files / str(pid)
            d.mkdir()
            (d / f"{pid}.hea").write_text(f"{pid} 12 100 1200\n", encoding="utf-8")
            (d / f"{pid}.dat").write_bytes(b"")
        if stray_ds_store:
            # The real release ships one of these, and it is even checksummed.
            (files / ".DS_Store").write_bytes(b"\x00\x01")

        lines = [f"files/{pid}/{pid}" for pid, *_ in rows]
        if duplicate_in_records and lines:
            lines.append(lines[0])  # RECORDS really does repeat one entry
        (tmp_path / "RECORDS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        from ecgbench.config import LabelConfig

        return replace(
            sample_config, slug="brugada_huca", metadata_csv="ecgbench_metadata.csv",
            record_id_column="patient_id", patient_id_column=None,
            signal_path_columns={100: "signal_path"}, default_sampling_rate=100,
            label_column="brugada", leads=12,
            labels=LabelConfig(source_csv="metadata.csv", join_column="patient_id"),
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.brugada_huca import BrugadaHUCASplitter

        assert isinstance(get_splitter("brugada_huca"), BrugadaHUCASplitter)

    def test_derives_signal_paths_and_caches_the_csv(self, sample_config, tmp_path):
        """The whole reason this splitter exists: metadata.csv has no path column."""
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(188981, 1, 1, 0), (251972, 0, 0, 0)])
        df = get_splitter("brugada_huca").load_metadata(tree, config)

        assert "signal_path" not in pd.read_csv(tree / "metadata.csv").columns
        paths = dict(zip(df["patient_id"], df["signal_path"]))
        assert paths == {188981: "files/188981/188981", 251972: "files/251972/251972"}
        # Written to disk because validate_dataset re-reads it and rebuilds paths.
        assert (tree / config.metadata_csv).exists()
        assert "signal_path" in pd.read_csv(tree / config.metadata_csv).columns

    def test_labels_are_carried_through(self, sample_config, tmp_path):
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(1, 0, 0, 0), (2, 1, 1, 1), (3, 2, 0, 0)])
        df = get_splitter("brugada_huca").load_metadata(tree, config)

        for col in ("brugada", "basal_pattern", "sudden_death"):
            assert col in df.columns
        assert df.set_index("patient_id")["brugada"].to_dict() == {1: 0, 2: 1, 3: 2}

    def test_stray_ds_store_is_not_mistaken_for_a_record(self, sample_config, tmp_path):
        """The release ships files/.DS_Store; globbing files/* must skip it."""
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(1, 0, 0, 0), (2, 1, 0, 0)], stray_ds_store=True)

        splitter = get_splitter("brugada_huca")
        df = splitter.load_metadata(tree, config)

        assert len(df) == 2
        assert ".DS_Store" not in set(df["patient_id"].astype(str))

    def test_duplicate_line_in_records_does_not_duplicate_a_row(
        self, sample_config, tmp_path
    ):
        """RECORDS lists one record twice; metadata.csv is the source of truth."""
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(1, 0, 0, 0), (2, 1, 0, 0)],
                          duplicate_in_records=True)
        records = (tree / "RECORDS").read_text().split()
        assert len(records) == 3 and len(set(records)) == 2  # the release's quirk

        df = get_splitter("brugada_huca").load_metadata(tree, config)

        assert len(df) == 2
        assert df["patient_id"].is_unique

    def test_subject_without_a_header_warns(self, sample_config, tmp_path, caplog):
        """A partial download should say so rather than fail every record later."""
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(1, 0, 0, 0), (2, 1, 0, 0)])
        (tree / "files" / "2" / "2.hea").unlink()

        with caplog.at_level("WARNING"):
            get_splitter("brugada_huca").load_metadata(tree, config)

        assert "no .hea on disk" in caplog.text

    def test_stratification_uses_brugada_verbatim(self, sample_config, tmp_path):
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [(i, i % 3, 0, 0) for i in range(1, 31)])
        splitter = get_splitter("brugada_huca")
        df = splitter.load_metadata(tree, config)

        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "brugada"
        # Verbatim: no pooling, no renaming, so nothing can drift from load_labels.
        assert labels.tolist() == df["brugada"].tolist()

    def test_rare_class_is_warned_about_but_kept(self, sample_config, tmp_path, caplog):
        """'other/atypical' must not be pooled — it is not interchangeable."""
        config = self._config(sample_config)
        rows = [(i, 0, 0, 0) for i in range(1, 21)] + [(99, 2, 0, 0)]
        splitter = get_splitter("brugada_huca")
        df = splitter.load_metadata(self._tree(tmp_path, rows), config)

        with caplog.at_level("WARNING"):
            labels = splitter.get_stratification_labels(df, config)

        assert set(labels) == {0, 2}          # class 2 survives
        assert "cannot appear in every fold" in caplog.text

    def test_missing_metadata_csv_names_the_expected_layout(self, sample_config, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata.csv"):
            get_splitter("brugada_huca").load_metadata(tmp_path, self._config(sample_config))


class TestLeipzigHeartCenterSplitter:
    """Leipzig's binding constraint is class count: 7 diagnoses over 39 records."""

    LAYOUTS = {
        "child": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4",
                  "V5", "V6", "ABL12", "RVA12", "CS12"],
        # The adult layout in the real x100 puts RVA12 last, after the CS channels.
        "adult": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4",
                  "V5", "V6", "ABL12", "CS12", "RVA12"],
    }

    def _tree(self, tmp_path, children, adults):
        """children/adults: [(subject_id, record, diagnosis)] -> a loadable tree."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        pd.DataFrame({
            "subject_id": [s for s, _, _ in children],
            "file_name": [r for _, r, _ in children],
            "gender": ["M"] * len(children),
            "age": ["10.0"] * len(children),
            "diagnosis": [d for _, _, d in children],
            "ap_loacation": [""] * len(children),
            "ecg_duration": ["0:00:02.0"] * len(children),
        }).to_csv(tmp_path / "children-subject-info.csv", index=False)
        pd.DataFrame({
            "subject_id": [s for s, _, _ in adults],
            "file_name": [r for _, r, _ in adults],
            "gender": ["F"] * len(adults),
            "age": ["50.0"] * len(adults),
            "diagnosis": [d for _, _, d in adults],
            "ecg_duration": ["0:00:02.0"] * len(adults),
        }).to_csv(tmp_path / "adults-subject-info.csv", index=False)

        for cohort, rows in (("child", children), ("adult", adults)):
            names = self.LAYOUTS[cohort]
            n = len(names)
            for _, record, _ in rows:
                signal = np.tile(
                    np.arange(1954, dtype=np.float64)[:, None] / 1000.0, (1, n)
                )
                wfdb.wrsamp(record, fs=977, units=["mV"] * n, sig_name=names,
                            p_signal=signal, fmt=["16"] * n, adc_gain=[2000.0] * n,
                            baseline=[0] * n, write_dir=str(tmp_path))
                wfdb.wrann(record, "atr", np.array([100, 200, 300]),
                           np.array(["N", "X", "+"]),
                           aux_note=["", "AVRT", "(N"], fs=977,
                           write_dir=str(tmp_path),
                           custom_labels=[(42, "X", "Tachycardias")])
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="leipzig_heart_center_ecg",
            metadata_csv="ecgbench_metadata.csv", record_id_column="record_name",
            patient_id_column=None, signal_path_columns={977: "signal_path"},
            default_sampling_rate=977, label_column="stratify_class", leads=12,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.leipzig_heart_center_ecg import (
            LeipzigHeartCenterSplitter,
        )

        assert isinstance(
            get_splitter("leipzig_heart_center_ecg"), LeipzigHeartCenterSplitter
        )

    def test_joins_the_two_subject_csvs_and_writes_the_metadata(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            children=[("001", "x001", "AVRT-WPW"), ("002", "x002", "AVNRT")],
            adults=[("100", "x100", "TOF with VT")],
        )
        df = get_splitter("leipzig_heart_center_ecg").load_metadata(tree, config)

        # One frame from two CSVs with different columns.
        assert len(df) == 3
        assert set(df["cohort"]) == {"child", "adult"}
        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["x001", "x002", "x100"]
        # Written to disk because validate_dataset re-reads it from there.
        assert (tree / config.metadata_csv).exists()

    def test_cached_metadata_keeps_zero_padded_subject_ids(self, sample_config, tmp_path):
        """subject_id is '001'/'0010' in the source; an int read would lose the padding."""
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            children=[("001", "x001", "AVRT"), ("0010", "x0010", "AVNRT")],
            adults=[],
        )
        splitter = get_splitter("leipzig_heart_center_ecg")

        first = splitter.load_metadata(tree, config)          # builds and writes
        second = splitter.load_metadata(tree, config)         # re-reads the cache

        assert sorted(first["subject_id"]) == ["001", "0010"]
        assert sorted(second["subject_id"]) == ["001", "0010"]
        assert list(first["record_name"]) == list(second["record_name"])

    def test_stratification_uses_the_family_not_the_full_diagnosis(self, sample_config, tmp_path):
        """7 diagnoses over 39 records cannot be spread across 10 folds; 3 can."""
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        # 10 of each family, so nothing pools — as in the real dataset.
        children = (
            [(f"{i:03d}", f"x{i:03d}", "AVRT-WPW") for i in range(1, 6)]
            + [(f"{i:03d}", f"x{i:03d}", "AVRT-PJRT") for i in range(6, 11)]
            + [(f"{i:03d}", f"x{i:03d}", "AVNRT") for i in range(11, 21)]
        )
        adults = (
            [(f"1{i:02d}", f"x1{i:02d}", "TOF with VT") for i in range(0, 9)]
            + [("109", "x109", "TOF without VT")]
        )
        splitter = get_splitter("leipzig_heart_center_ecg")
        df = splitter.load_metadata(self._tree(tmp_path, children, adults), config)

        labels = splitter.get_stratification_labels(df, config)

        # Five shipped diagnoses collapse to three families, none pooled.
        assert df["diagnosis"].nunique() == 5
        assert labels.value_counts().to_dict() == {"AVRT": 10, "AVNRT": 10, "TOF": 10}
        assert "OTHER" not in set(labels)

    def test_missing_stratify_column_is_refused(self, sample_config):
        config = self._config(sample_config)
        splitter = get_splitter("leipzig_heart_center_ecg")

        with pytest.raises(ValueError, match="stratify_class"):
            splitter.get_stratification_labels(
                pd.DataFrame({"record_name": ["x001"], "diagnosis": ["AVRT"]}), config
            )

    def test_folds_are_stratified_and_no_record_repeats(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        children = (
            [(f"{i:03d}", f"x{i:03d}", "AVRT") for i in range(1, 11)]
            + [(f"{i:03d}", f"x{i:03d}", "AVNRT") for i in range(11, 21)]
        )
        adults = [(f"1{i:02d}", f"x1{i:02d}", "TOF with VT") for i in range(0, 10)]
        splitter = get_splitter("leipzig_heart_center_ecg")
        df = splitter.load_metadata(self._tree(tmp_path, children, adults), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assert len(result.folds) == 5
        # One record per subject, so a record-level split is already subject-level.
        assert config.patient_id_column is None
        seen = pd.concat(result.folds.values())[config.record_id_column]
        assert len(seen) == len(df) and seen.is_unique
        # Every fold carries all three families — the point of stratifying.
        for fold in result.folds.values():
            families = set(fold["stratify_class"])
            assert families == {"AVRT", "AVNRT", "TOF"}, families


class TestNorwegianAthleteECGSplitter:
    """28 header-labelled records, stratified on the cardiologist's opening rhythm."""

    RHYTHMS = ("Normal sinus rhythm", "Sinus arrhythmia", "Sinus bradycardia")

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="norwegian_athlete_ecg",
            metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={500: "signal_path"},
            label_column="cardiologist_primary_rhythm",
            url="https://physionet.org/content/norwegian-athlete-ecg/1.0.0/",
        )

    def _source(self, tmp_path, n=28):
        """Write n headers whose rhythms repeat the real 16/7/5-ish proportions."""
        names = [f"ath_{i:03d}" for i in range(1, n + 1)]
        (tmp_path / "RECORDS").write_text("\n".join(names) + "\n", encoding="utf-8")
        for index, name in enumerate(names):
            rhythm = self.RHYTHMS[0] if index % 4 else self.RHYTHMS[1 + index % 2]
            (tmp_path / f"{name}.hea").write_text(
                f"{name} 12 500 5000\n"
                f"{name}.dat 16 50000/mV 16 0 100 200 0 I\n"
                f"#SL12: Sinus bradycardia, Right axis deviation, Borderline ECG\n"
                f"#C: {rhythm}, Normal ECG\n",
                encoding="utf-8",
            )
        return tmp_path

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.norwegian_athlete_ecg import (
            NorwegianAthleteECGSplitter,
        )

        assert isinstance(get_splitter("norwegian_athlete_ecg"), NorwegianAthleteECGSplitter)

    def test_load_metadata_builds_and_caches_the_csv(self, sample_config, tmp_path):
        """The release ships no metadata table, and validate_dataset re-reads it
        from disk — so writing the cache is load-bearing, not an optimisation."""
        config = self._config(sample_config)
        splitter = get_splitter("norwegian_athlete_ecg")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert len(df) == 28
        assert (tmp_path / "ecgbench_metadata.csv").exists()
        assert df["record_name"].tolist()[:2] == ["ath_001", "ath_002"]
        # Signals sit flat in the root, named by the bare stem.
        assert df["signal_path"].tolist() == df["record_name"].tolist()

    def test_cached_csv_is_reused_on_the_second_call(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("norwegian_athlete_ecg")
        path = self._source(tmp_path)

        splitter.load_metadata(path, config)
        (path / "ath_001.hea").unlink()  # cache must be read instead of the headers
        df = splitter.load_metadata(path, config)

        assert len(df) == 28

    def test_list_columns_round_trip_without_a_comma(self, sample_config, tmp_path):
        """Findings hold commas of their own, so the CSV must not join them on one."""
        from ecgbench.splitting.strategies.norwegian_athlete_ecg import LIST_SEPARATOR

        config = self._config(sample_config)
        splitter = get_splitter("norwegian_athlete_ecg")
        splitter.load_metadata(self._source(tmp_path), config)

        reread = pd.read_csv(tmp_path / "ecgbench_metadata.csv")
        assert LIST_SEPARATOR == ";"
        assert reread.loc[0, "sl12_findings"] == (
            "Sinus bradycardia;Right axis deviation"
        )

    def test_stratification_labels_come_from_the_label_loader(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("norwegian_athlete_ecg")
        df = splitter.load_metadata(self._source(tmp_path), config)

        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "rhythm"
        assert set(labels) <= set(self.RHYTHMS)
        assert len(labels) == 28

    def test_stratification_requires_load_metadata_first(self, sample_config):
        splitter = get_splitter("norwegian_athlete_ecg")

        with pytest.raises(ValueError, match="cardiologist_primary_rhythm"):
            splitter.get_stratification_labels(pd.DataFrame({"record_name": ["a"]}),
                                               self._config(sample_config))

    def test_folds_are_record_disjoint_and_cover_every_record(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("norwegian_athlete_ecg")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assert len(result.folds) == 5
        # One record per athlete, so a record-level split is already athlete-level.
        assert config.patient_id_column is None
        seen = pd.concat(result.folds.values())[config.record_id_column]
        assert len(seen) == len(df) and seen.is_unique


class TestMHDEffectECGMRISplitter:
    """Folds must group by the DERIVED subject key, not the per-scanner number."""

    def _header(self, record, *, field="3T", position="Feet first (Ff)",
                sex="Male", age="27years", weight="75kg", height="190cm"):
        return (
            f"{record} 3 1024 25000\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 I\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 II\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 III\n"
            f"#--Magnetic field strength:{field}\n"
            "#--MR scanner:Siemens Magnetom Skyra\n"
            "#--Orientation of the static magnetic field (B0):Horizontal\n"
            "#--ECG recorder:MIPM Tesla M3 Patient Monitor\n"
            "#--ADC resolution:24bit\n"
            "#--ADC input voltage range:+/-2.4mV\n"
            "#--ECG lead configuration:Reduced Einthoven Triangle\n"
            f"#--Sex:{sex}\n"
            f"#--Age:{age}\n"
            f"#--Weight:{weight}\n"
            f"#--Height:{height}\n"
            f"#--Positon in the scanner:{position}\n"
            "#--Respiration:Spontaneous respiration\n"
        )

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="mhd_effect_ecg_mri",
            metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="subject_key",
            default_sampling_rate=1024, sampling_rates=[1024],
            signal_path_columns={1024: "signal_path"}, label_column="condition",
            url="https://physionet.org/content/mhd-effect-ecg-mri/1.0.0/",
        )

    def _source(self, tmp_path, n_subjects=12):
        """One 3T + one reference record per subject, plus a cross-scanner pair.

        Subject 01 is deliberately recorded at both 3T and 7T under the SAME
        demographics, mirroring the real 3T01 == 7T04 case.
        """
        records = {}
        for i in range(1, n_subjects + 1):
            demo = dict(age=f"{20 + i}years", weight=f"{60 + i}kg", height=f"{170 + i}cm",
                        sex="Female" if i % 3 == 0 else "Male")
            records[f"ECGMRI3T{i:02d}Ff"] = self._header(f"ECGMRI3T{i:02d}Ff", **demo)
            records[f"ECGMRI3T{i:02d}Out"] = self._header(
                f"ECGMRI3T{i:02d}Out", field="Outside the scanner",
                position="Outside the scanner", **demo,
            )
            if i == 1:
                # Same person, different scanner and different subject slot.
                records["ECGMRI7T09Ff"] = self._header(
                    "ECGMRI7T09Ff", field="7T", **demo
                )
        (tmp_path / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
        for record, text in records.items():
            (tmp_path / f"{record}.hea").write_text(text, encoding="utf-8")
        return tmp_path

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.mhd_effect_ecg_mri import MHDEffectECGMRISplitter

        assert isinstance(get_splitter("mhd_effect_ecg_mri"), MHDEffectECGMRISplitter)

    def test_load_metadata_builds_and_caches_the_csv(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("mhd_effect_ecg_mri")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert len(df) == 25  # 12 subjects x 2 records + 1 cross-scanner record
        assert (tmp_path / "ecgbench_metadata.csv").exists()
        assert df["signal_path"].tolist() == df["record_name"].tolist()
        assert {"condition", "subject_key", "n_qrs"} <= set(df.columns)

    def test_cached_csv_keeps_subject_number_a_string(self, sample_config, tmp_path):
        """'01' must not come back as int 1 — it would stop matching the filename."""
        config = self._config(sample_config)
        splitter = get_splitter("mhd_effect_ecg_mri")
        path = self._source(tmp_path)

        splitter.load_metadata(path, config)
        (path / "ECGMRI3T01Ff.hea").unlink()   # force the cache path
        df = splitter.load_metadata(path, config)

        assert df["subject_number"].map(type).eq(str).all()
        assert "01" in set(df["subject_number"])

    def test_stratification_labels_come_from_the_label_loader(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("mhd_effect_ecg_mri")
        df = splitter.load_metadata(self._source(tmp_path), config)

        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "condition"
        assert set(labels) == {"3T", "7T", "reference"}

    def test_stratification_requires_the_subject_column(self, sample_config):
        """Without it, split_dataset would silently fall back to ungrouped folds."""
        splitter = get_splitter("mhd_effect_ecg_mri")
        df = pd.DataFrame({"record_name": ["ECGMRI3T01Ff"], "condition": ["3T"]})

        with pytest.raises(ValueError, match="subject_key"):
            splitter.get_stratification_labels(df, self._config(sample_config))

    def test_no_subject_spans_a_fold(self, sample_config, tmp_path):
        """The whole point of the derived key: one person, one fold.

        Subject 01 appears as slots 3T01 and 7T09, so a filename-based key would
        scatter them. Folds must keep every record of one subject_key together.
        """
        config = self._config(sample_config)
        splitter = get_splitter("mhd_effect_ecg_mri")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assigned = pd.concat(
            [fold.assign(fold_id=n) for n, fold in result.folds.items()]
        )
        per_subject = assigned.groupby("subject_key")["fold_id"].nunique()
        assert (per_subject == 1).all(), per_subject[per_subject > 1].to_dict()

        # And specifically the cross-scanner subject.
        cross = assigned[assigned["scanner_subject_slot"].isin(["3T01", "7T09"])]
        assert cross["subject_key"].nunique() == 1
        assert cross["fold_id"].nunique() == 1
        assert len(cross) == 3

        seen = assigned[config.record_id_column]
        assert len(seen) == len(df) and seen.is_unique


class TestWCTECGSplitter:
    """540 segments from 92 patients, so patient grouping is not optional."""

    DIAGNOSES = (
        ("Non ST\xa0segment\xa0elevation myocardial infarction (NSTEMI)",
         "Myocardial infarction"),
        ("Stable angina", "Angina or coronary artery disease"),
        ("Atrial fibrillation", "Atrial fibrillation or flutter"),
        ("not reported", "Not reported"),
    )

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="wctecgdb",
            metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="patient_id",
            signal_path_columns={800: "signal_path"},
            default_sampling_rate=800, sampling_rates=[800],
            label_column="diagnosis_group",
            url="https://physionet.org/content/wctecgdb/1.0.1/",
        )

    def _source(self, tmp_path, n_patients=40, segments=3):
        """Write n_patients x segments headers, cycling the four diagnoses."""
        names = []
        for patient in range(1, n_patients + 1):
            diagnosis = self.DIAGNOSES[patient % len(self.DIAGNOSES)][0]
            for segment in range(1, segments + 1):
                name = f"patient{patient:03d}/seg{segment:02d}"
                names.append(name)
                path = tmp_path / f"{name}.hea"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"seg{segment:02d} 37 800 8001\n"
                    f"seg{segment:02d}.dat 16 36213.4604(-6137)/mV 0 0 500 -11346 0 I-Raw\n"
                    "\n"
                    f"#Age: {40 + patient}\n"
                    f"#Sex: {'M' if patient % 3 else 'F'}\n"
                    f"#Diagnosis report: {diagnosis}\n",
                    encoding="cp1252",
                )
        (tmp_path / "RECORDS").write_text("\n".join(names) + "\n", encoding="utf-8")
        return tmp_path

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.wctecgdb import WCTECGSplitter

        assert isinstance(get_splitter("wctecgdb"), WCTECGSplitter)

    def test_load_metadata_builds_and_caches_the_csv(self, sample_config, tmp_path):
        """The release ships no metadata table, and validate_dataset re-reads it
        from disk — so writing the cache is load-bearing, not an optimisation."""
        config = self._config(sample_config)
        splitter = get_splitter("wctecgdb")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert len(df) == 120
        assert (tmp_path / "ecgbench_metadata.csv").exists()
        assert df["record_name"].tolist()[:2] == ["patient001_seg01", "patient001_seg02"]

    def test_signal_path_keeps_the_directory_the_record_id_flattens(
        self, sample_config, tmp_path
    ):
        """patient001_seg01 identifies the record; patient001/seg01 locates it."""
        config = self._config(sample_config)
        splitter = get_splitter("wctecgdb")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert df["signal_path"].tolist()[:2] == [
            "patient001/seg01", "patient001/seg02",
        ]
        assert (tmp_path / (df["signal_path"][0] + ".hea")).exists()

    def test_cached_csv_is_reused_on_the_second_call(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("wctecgdb")
        path = self._source(tmp_path)

        splitter.load_metadata(path, config)
        (path / "patient001/seg01.hea").unlink()  # cache, not the headers
        df = splitter.load_metadata(path, config)

        assert len(df) == 120

    def test_list_column_round_trips_without_a_comma(self, sample_config, tmp_path):
        """The channel list holds commas itself ("V1, V1-raw"), so ';' it is."""
        from ecgbench.splitting.strategies.wctecgdb import LIST_SEPARATOR

        config = self._config(sample_config)
        splitter = get_splitter("wctecgdb")
        path = self._source(tmp_path)
        header = (path / "patient001/seg01.hea").read_text(encoding="cp1252")
        (path / "patient001/seg01.hea").write_text(
            header + "#Reconstruct Precordials: V1, V1-raw, V2, V2-raw\n",
            encoding="cp1252",
        )
        splitter.load_metadata(path, config)

        reread = pd.read_csv(tmp_path / "ecgbench_metadata.csv")
        assert LIST_SEPARATOR == ";"
        assert reread.loc[0, "reconstructed_precordials"] == "V1;V1-raw;V2;V2-raw"

    def test_stratification_labels_come_from_the_label_loader(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("wctecgdb")
        df = splitter.load_metadata(self._source(tmp_path), config)

        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "diagnosis_group"
        assert set(labels) == {group for _, group in self.DIAGNOSES}
        assert len(labels) == 120

    def test_stratification_requires_load_metadata_first(self, sample_config):
        splitter = get_splitter("wctecgdb")

        with pytest.raises(ValueError, match="diagnosis_group"):
            splitter.get_stratification_labels(
                pd.DataFrame({"record_name": ["a"]}), self._config(sample_config)
            )

    def test_no_patient_spans_two_folds(self, sample_config, tmp_path):
        """The whole point of grouping: one patient contributes up to 31 near-
        duplicate segments, and its diagnosis label is constant across them."""
        config = self._config(sample_config)
        splitter = get_splitter("wctecgdb")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assert len(result.folds) == 5
        assert result.group_column == "patient_id"
        fold_of = {}
        for fold, frame in result.folds.items():
            for patient in frame["patient_id"]:
                assert fold_of.setdefault(patient, fold) == fold
        assert len(fold_of) == 40

        seen = pd.concat(result.folds.values())[config.record_id_column]
        assert len(seen) == len(df) and seen.is_unique


class TestECGCIPASplitter:
    """5,749 records from 60 subjects, in near-duplicate triplicates."""

    TREATMENTS = (
        "Ranolazine", "Verapamil", "Chloroquine", "Lopinavir+Ritonavir",
        "Placebo", "Dofetilide",
    )

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ecgcipa",
            metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_id", patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"},
            default_sampling_rate=1000, sampling_rates=[1000],
            label_column="treatment",
            url="https://physionet.org/content/ecgcipa/1.0.0/",
        )

    def _source(self, tmp_path, n_subjects=30, timepoints=4, replicates=3):
        """Write RECORDS plus the four analysis CSVs for a miniature study.

        Mirrors the release's shape rather than its size: one treatment per
        subject, `timepoints` nominal timepoints, `replicates` records each.
        """
        stems, adeg, adpc, adsl, addm = [], [], [], [], []
        for index in range(n_subjects):
            subject = 1001 + index
            treatment = self.TREATMENTS[index % len(self.TREATMENTS)]
            adsl.append({
                "USUBJID": subject, "AGE": 25 + index, "SEX": "M" if index % 2 else "F",
                "RACE": "  WHITE", "ETHNIC": "NOT HISPANIC OR LATINO",
                "ARM": treatment, "ACTARM": treatment,
            })
            for code, value in (("HEIGHT", 175.0), ("WEIGHT", 70.0), ("BMI", 22.9),
                                ("SYSBP", 120.0), ("DIABP", 75.0)):
                addm.append({"USUBJID": subject, "PARAMCD": code, "AVAL": value})
            for timepoint in range(1, timepoints + 1):
                adpc.append({
                    "USUBJID": subject, "APERIOD": 1, "ATPTN": timepoint,
                    "PARAMCD": "RAN", "AVAL": 100.0 * timepoint, "LLOQFL": None,
                })
                for replicate in range(1, replicates + 1):
                    record = f"{subject}-{timepoint:02d}-{replicate}"
                    stems.append(f"{subject}/{record}")
                    for param, value in (("HR", 64.0), ("RR", 940.0), ("PR", 188.0),
                                         ("QRS", 78.0), ("QT", 371.0), ("QTCF", 378.0),
                                         ("JTP", 232.0), ("JTPC", 240.0),
                                         ("TPTE", 61.0)):
                        adeg.append({
                            "STUDYID": "SCR-004", "USUBJID": subject,
                            "TRTA": treatment, "TRTP": treatment,
                            "TRTSEQA": treatment, "APERIOD": 1,
                            "APERIODC": "Period 1", "ATPT": f"{timepoint} hrs",
                            "ATPTN": timepoint, "NRRLT": float(timepoint),
                            "ARRLT": float(timepoint), "ADY": 1, "APERDAY": 1,
                            "ADTM": "2017-03-27 06:57:10", "AEGBLFL": None,
                            "ECGPCFL": "Y", "EGREFID": record, "DTYPE": None,
                            "PARAMCD": param, "AVAL": value,
                            "EGREPNUM": replicate,
                        })

        lines = [f"raw/{stem}" for stem in stems] + [f"medians/{stem}" for stem in stems]
        (tmp_path / "RECORDS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        pd.DataFrame(adeg).to_csv(tmp_path / "adeg.csv", index=False)
        pd.DataFrame(adpc).to_csv(tmp_path / "adpc.csv", index=False)
        pd.DataFrame(adsl).to_csv(tmp_path / "adsl.csv", index=False)
        pd.DataFrame(addm).to_csv(tmp_path / "addm.csv", index=False)
        return tmp_path

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.ecgcipa import ECGCIPASplitter

        assert isinstance(get_splitter("ecgcipa"), ECGCIPASplitter)

    def test_load_metadata_builds_and_caches_the_csv(self, sample_config, tmp_path):
        """The release ships no record-to-file table, and validate_dataset re-reads
        the CSV from disk — so writing the cache is load-bearing."""
        config = self._config(sample_config)
        splitter = get_splitter("ecgcipa")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert len(df) == 30 * 4 * 3
        assert (tmp_path / "ecgbench_metadata.csv").exists()
        assert df["patient_id"].nunique() == 30

    def test_signal_path_points_at_raw_not_the_derived_medians(
        self, sample_config, tmp_path
    ):
        """Both directories hold a record per id; only raw/ is the ECG."""
        config = self._config(sample_config)
        splitter = get_splitter("ecgcipa")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert df["signal_path"].str.startswith("raw/").all()
        assert df["median_beat_path"].str.startswith("medians/").all()
        assert df.loc[0, "signal_path"] == "raw/1001/1001-01-1"

    def test_cached_csv_is_reused_on_the_second_call(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("ecgcipa")
        path = self._source(tmp_path)

        splitter.load_metadata(path, config)
        (path / "adeg.csv").unlink()  # the cache is read, not the source tables
        df = splitter.load_metadata(path, config)

        assert len(df) == 30 * 4 * 3

    def test_stratification_labels_come_from_the_label_loader(
        self, sample_config, tmp_path
    ):
        config = self._config(sample_config)
        splitter = get_splitter("ecgcipa")
        df = splitter.load_metadata(self._source(tmp_path), config)

        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "treatment"
        assert set(labels) == set(self.TREATMENTS)
        assert len(labels) == len(df)

    def test_stratification_requires_load_metadata_first(self, sample_config):
        splitter = get_splitter("ecgcipa")

        with pytest.raises(ValueError, match="treatment"):
            splitter.get_stratification_labels(
                pd.DataFrame({"record_id": ["a"]}), self._config(sample_config)
            )

    def test_no_patient_spans_two_folds(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("ecgcipa")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assert len(result.folds) == 5
        assert result.group_column == "patient_id"
        fold_of = {}
        for fold, frame in result.folds.items():
            for patient in frame["patient_id"]:
                assert fold_of.setdefault(patient, fold) == fold
        assert len(fold_of) == 30

        seen = pd.concat(result.folds.values())[config.record_id_column]
        assert len(seen) == len(df) and seen.is_unique

    def test_triplicates_of_a_timepoint_never_split(self, sample_config, tmp_path):
        """The three records of a timepoint are the same person seconds apart at
        the same plasma concentration — near-duplicates, not independent samples.

        Patient grouping is what keeps them together; nothing else would, and a
        per-record split would put a near-copy of most test records in train.
        """
        config = self._config(sample_config)
        splitter = get_splitter("ecgcipa")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        fold_of = {}
        for fold, frame in result.folds.items():
            for row in frame.itertuples():
                key = (row.patient_id, row.period, row.timepoint_n)
                assert fold_of.setdefault(key, fold) == fold
        assert len(fold_of) == 30 * 4


class TestECGDMMLDSplitter:
    """4,211 records from 22 subjects, in near-duplicate triplicates."""

    #: ARMCD code -> treatment, per the shipped column description.
    ARMS = {
        "A": "Dofetilide",
        "B": "Lidocaine + Dofetilide",
        "C": "Mexiletine + Dofetilide",
        "D": "Moxifloxacin + Diltiazem",
        "E": "Placebo",
    }

    #: The two randomised sequences used by the miniature study below. Real
    #: v1.0.0 has ten, 2-3 subjects each.
    SEQUENCES = ("E-A-B-D-C", "C-B-D-E-A")

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ecgdmmld",
            metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_id", patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"},
            default_sampling_rate=1000, sampling_rates=[1000],
            label_column="treatment",
            url="https://physionet.org/content/ecgdmmld/1.0.0/",
        )

    def _source(self, tmp_path, n_subjects=20, periods=5, timepoints=4, replicates=3):
        """Write a miniature SCR-003.Clinical.Data.csv.

        Mirrors the release's shape rather than its size: a complete crossover in
        which every subject passes through all five arms, `timepoints` nominal
        timepoints per period and `replicates` records each.
        """
        rows = []
        for index in range(n_subjects):
            subject = 2001 + index
            sequence = self.SEQUENCES[index % len(self.SEQUENCES)]
            for period in range(1, periods + 1):
                treatment = self.ARMS[sequence.split("-")[period - 1]]
                for timepoint_index in range(timepoints):
                    # -0.5 is the pre-dose triplicate; the rest are post-dose.
                    timepoint = -0.5 if timepoint_index == 0 else float(timepoint_index)
                    for replicate in range(1, replicates + 1):
                        rows.append({
                            "EGREFID": f"{subject}-{period}-{timepoint_index}-{replicate}",
                            "RANDID": subject,
                            "SEX": "M" if index % 2 else "F",
                            "AGE": 21 + index,
                            "HGHT": 175.0, "WGHT": 70.0,
                            "SYSBP": 120.0, "DIABP": 75.0,
                            "RACE": "WHITE", "ETHNIC": "NOT HISPANIC OR LATINO",
                            "ARMCD": sequence,
                            "VISIT": f"PERIOD-{period}-DOSING",
                            "TRTA": treatment,
                            "DOF": None, "LIDO": None, "MEXI": None,
                            "MOXI": None, "MOXI.M2": None, "DILT": None,
                            "TPT": timepoint,
                            "BASELINE": "Y" if timepoint_index == 0 else "N",
                            "RR": 1000.0, "PR": 166.0, "QT": 420.0, "QRS": 72.0,
                            "JTPEAK": 263.0, "TPEAKTEND": 85.0,
                            "TPEAKTPEAKP": None,
                            "ERD_30": 52.0, "LRD_30": 28.0,
                            "Twave_amplitude": 727.78, "Twave_asymmetry": 0.19,
                            "Twave_flatness": 0.53,
                        })
        pd.DataFrame(rows).to_csv(
            tmp_path / "SCR-003.Clinical.Data.csv", index=False
        )
        return tmp_path

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.ecgdmmld import ECGDMMLDSplitter

        assert isinstance(get_splitter("ecgdmmld"), ECGDMMLDSplitter)

    def test_load_metadata_builds_and_caches_the_csv(self, sample_config, tmp_path):
        """The shipped table has no signal-path column, and validate_dataset
        re-reads the CSV from disk — so writing the cache is load-bearing."""
        config = self._config(sample_config)
        splitter = get_splitter("ecgdmmld")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert len(df) == 20 * 5 * 4 * 3
        assert (tmp_path / "ecgbench_metadata.csv").exists()
        assert df["patient_id"].nunique() == 20

    def test_signal_path_points_at_raw_not_the_derived_medians(
        self, sample_config, tmp_path
    ):
        """Both directories hold a record per id; only raw/ is the ECG."""
        config = self._config(sample_config)
        splitter = get_splitter("ecgdmmld")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert df["signal_path"].str.startswith("raw/").all()
        assert df["median_beat_path"].str.startswith("medians/").all()
        assert df.loc[0, "signal_path"] == "raw/2001/2001-1-0-1"

    def test_cached_csv_is_reused_on_the_second_call(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("ecgdmmld")
        path = self._source(tmp_path)

        splitter.load_metadata(path, config)
        (path / "SCR-003.Clinical.Data.csv").unlink()  # cache read, not the source
        df = splitter.load_metadata(path, config)

        assert len(df) == 20 * 5 * 4 * 3

    def test_stratification_labels_come_from_the_label_loader(
        self, sample_config, tmp_path
    ):
        config = self._config(sample_config)
        splitter = get_splitter("ecgdmmld")
        df = splitter.load_metadata(self._source(tmp_path), config)

        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "treatment"
        assert set(labels) == set(self.ARMS.values())
        assert len(labels) == len(df)

    def test_stratification_requires_load_metadata_first(self, sample_config):
        splitter = get_splitter("ecgdmmld")

        with pytest.raises(ValueError, match="treatment"):
            splitter.get_stratification_labels(
                pd.DataFrame({"record_id": ["a"]}), self._config(sample_config)
            )

    def test_no_patient_spans_two_folds(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("ecgdmmld")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assert len(result.folds) == 5
        assert result.group_column == "patient_id"
        fold_of = {}
        for fold, frame in result.folds.items():
            for patient in frame["patient_id"]:
                assert fold_of.setdefault(patient, fold) == fold
        assert len(fold_of) == 20

        seen = pd.concat(result.folds.values())[config.record_id_column]
        assert len(seen) == len(df) and seen.is_unique

    def test_triplicates_of_a_timepoint_never_split(self, sample_config, tmp_path):
        """The three records of a timepoint are the same person seconds apart at
        the same plasma concentration — near-duplicates, not independent samples.

        Patient grouping is what keeps them together; nothing else would, and a
        per-record split would put a near-copy of most test records in train.
        """
        config = self._config(sample_config)
        splitter = get_splitter("ecgdmmld")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        fold_of = {}
        for fold, frame in result.folds.items():
            for row in frame.itertuples():
                key = (row.patient_id, row.period, row.timepoint_hours)
                assert fold_of.setdefault(key, fold) == fold
        assert len(fold_of) == 20 * 5 * 4

    def test_a_complete_crossover_puts_every_arm_in_every_fold(
        self, sample_config, tmp_path
    ):
        """Every subject carries all five arms, so patient grouping alone gives
        each fold all five — the stratifier cannot separate them and does not try.

        This is the property that makes `treatment` a weak stratification target
        here, and it is worth pinning: a future change that started splitting by
        something other than patient would break it.
        """
        config = self._config(sample_config)
        splitter = get_splitter("ecgdmmld")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        for fold, frame in result.folds.items():
            assert set(frame["treatment"]) == set(self.ARMS.values()), fold


class TestECGRDVQSplitter:
    """5,232 records from 22 subjects, in exact triplicates."""

    #: ARMCD code -> treatment, per the shipped column description. The same five
    #: letters mean different drugs in the sibling SCR-003 release.
    ARMS = {
        "A": "Ranolazine",
        "B": "Dofetilide",
        "C": "Verapamil HCL",
        "D": "Quinidine Sulph",
        "E": "Placebo",
    }

    #: Dose per treatment — dofetilide is micrograms, the rest milligrams.
    DOSES = {
        "Ranolazine": (1500.0, "mg"),
        "Dofetilide": (500.0, "ug"),
        "Verapamil HCL": (120.0, "mg"),
        "Quinidine Sulph": (400.0, "mg"),
        "Placebo": (None, None),
    }

    #: Two of the eleven randomised sequences in v1.0.0, which uses 2-3 subjects
    #: each. Comma-separated, unlike ecgdmmld's dashes.
    SEQUENCES = ("A,C,E,D,B", "C,D,A,B,E")

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ecgrdvq",
            metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_id", patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"},
            default_sampling_rate=1000, sampling_rates=[1000],
            label_column="treatment",
            url="https://physionet.org/content/ecgrdvq/1.0.0/",
        )

    def _source(self, tmp_path, n_subjects=20, periods=5, timepoints=4, replicates=3):
        """Write a miniature SCR-002.Clinical.Data.csv.

        Mirrors the release's shape rather than its size: a complete crossover in
        which every subject passes through all five single-agent arms, `timepoints`
        nominal timepoints per period and `replicates` records each.
        """
        rows = []
        for index in range(n_subjects):
            subject = 1001 + index
            sequence = self.SEQUENCES[index % len(self.SEQUENCES)]
            for period in range(1, periods + 1):
                treatment = self.ARMS[sequence.split(",")[period - 1]]
                dose, dose_unit = self.DOSES[treatment]
                for timepoint_index in range(timepoints):
                    # -0.5 is the pre-dose triplicate; the rest are post-dose.
                    timepoint = -0.5 if timepoint_index == 0 else float(timepoint_index)
                    predose = timepoint_index == 0
                    for replicate in range(1, replicates + 1):
                        rows.append({
                            "EGREFID": f"{subject}-{period}-{timepoint_index}-{replicate}",
                            "RANDID": subject,
                            "SEX": "M" if index % 2 else "F",
                            "AGE": 21 + index,
                            "HGHT": 175.0, "WGHT": 70.0,
                            "SYSBP": 120.0, "DIABP": 75.0,
                            "RACE": "WHITE", "ETHNIC": "NOT HISPANIC OR LATINO",
                            "ARMCD": sequence,
                            "VISIT": f"PERIOD-{period}-DOSING",
                            "EXTRT": treatment,
                            "EXDOSE": dose, "EXDOSU": dose_unit,
                            "TPT": timepoint,
                            "BASELINE": "Y" if predose else "N",
                            # No PK sample pre-dose, and none at all on placebo.
                            "PCTEST": (
                                None if predose or treatment == "Placebo"
                                else treatment.split()[0]
                            ),
                            "PCSTRESN": (
                                None if predose or treatment == "Placebo" else 100.0
                            ),
                            "PCSTRESU": (
                                None if predose or treatment == "Placebo"
                                else ("pg/mL" if treatment == "Dofetilide" else "ng/mL")
                            ),
                            "RR": 1000.0, "PR": 166.0, "QT": 420.0, "QRS": 72.0,
                            "JTPEAK": 263.0, "TPEAKTEND": 85.0,
                            "TPEAKTPEAKP": None,
                            "ERD_30": 52.0, "LRD_30": 28.0,
                            "Twave_amplitude": 727.78, "Twave_asymmetry": 0.19,
                            "Twave_flatness": 0.53,
                        })
        pd.DataFrame(rows).to_csv(
            tmp_path / "SCR-002.Clinical.Data.csv", index=False
        )
        return tmp_path

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.ecgrdvq import ECGRDVQSplitter

        assert isinstance(get_splitter("ecgrdvq"), ECGRDVQSplitter)

    def test_load_metadata_builds_and_caches_the_csv(self, sample_config, tmp_path):
        """The shipped table has no signal-path column, and validate_dataset
        re-reads the CSV from disk — so writing the cache is load-bearing."""
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert len(df) == 20 * 5 * 4 * 3
        assert (tmp_path / "ecgbench_metadata.csv").exists()
        assert df["patient_id"].nunique() == 20

    def test_signal_path_points_at_raw_not_the_derived_medians(
        self, sample_config, tmp_path
    ):
        """Both directories hold a record per id; only raw/ is the ECG."""
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")

        df = splitter.load_metadata(self._source(tmp_path), config)

        assert df["signal_path"].str.startswith("raw/").all()
        assert df["median_beat_path"].str.startswith("medians/").all()
        assert df.loc[0, "signal_path"] == "raw/1001/1001-1-0-1"

    def test_cached_csv_is_reused_on_the_second_call(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")
        path = self._source(tmp_path)

        splitter.load_metadata(path, config)
        (path / "SCR-002.Clinical.Data.csv").unlink()  # cache read, not the source
        df = splitter.load_metadata(path, config)

        assert len(df) == 20 * 5 * 4 * 3

    def test_stratification_labels_come_from_the_label_loader(
        self, sample_config, tmp_path
    ):
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")
        df = splitter.load_metadata(self._source(tmp_path), config)

        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "treatment"
        assert set(labels) == set(self.ARMS.values())
        assert len(labels) == len(df)

    def test_stratification_requires_load_metadata_first(self, sample_config):
        splitter = get_splitter("ecgrdvq")

        with pytest.raises(ValueError, match="treatment"):
            splitter.get_stratification_labels(
                pd.DataFrame({"record_id": ["a"]}), self._config(sample_config)
            )

    def test_no_patient_spans_two_folds(self, sample_config, tmp_path):
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assert len(result.folds) == 5
        assert result.group_column == "patient_id"
        fold_of = {}
        for fold, frame in result.folds.items():
            for patient in frame["patient_id"]:
                assert fold_of.setdefault(patient, fold) == fold
        assert len(fold_of) == 20

        seen = pd.concat(result.folds.values())[config.record_id_column]
        assert len(seen) == len(df) and seen.is_unique

    def test_triplicates_of_a_timepoint_never_split(self, sample_config, tmp_path):
        """The three records of a timepoint are the same person seconds apart at the
        same plasma concentration — near-duplicates, not independent samples. In this
        release the structure is exact: all 1,744 groups hold precisely 3 records.

        Patient grouping is what keeps them together; nothing else would, and a
        per-record split would put a near-copy of most test records in train.
        """
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        fold_of = {}
        for fold, frame in result.folds.items():
            for row in frame.itertuples():
                key = (row.patient_id, row.period, row.timepoint_hours)
                assert fold_of.setdefault(key, fold) == fold
        assert len(fold_of) == 20 * 5 * 4

    def test_a_complete_crossover_puts_every_arm_in_every_fold(
        self, sample_config, tmp_path
    ):
        """Every subject carries all five arms, so patient grouping alone gives each
        fold all five — the stratifier cannot separate them and does not try.

        This is the property that makes `treatment` a weak stratification target
        here, and it is worth pinning: a future change that started splitting by
        something other than patient would break it.
        """
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")
        df = splitter.load_metadata(self._source(tmp_path), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        for fold, frame in result.folds.items():
            assert set(frame["treatment"]) == set(self.ARMS.values()), fold

    def test_an_early_withdrawal_still_lands_in_exactly_one_fold(
        self, sample_config, tmp_path
    ):
        """Subject 1002 completed 4 of the 5 periods, so its treatment mix is
        lopsided — placing it is the one thing stratifying on treatment buys here."""
        config = self._config(sample_config)
        splitter = get_splitter("ecgrdvq")
        path = self._source(tmp_path)

        # Drop the last period of one subject, as the release does for 1002.
        source = pd.read_csv(path / "SCR-002.Clinical.Data.csv")
        withdrawn = (source.RANDID == 1002) & (source.VISIT == "PERIOD-5-DOSING")
        source[~withdrawn].to_csv(path / "SCR-002.Clinical.Data.csv", index=False)

        df = splitter.load_metadata(path, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=5)

        assert df[df.patient_id == "1002"].period.nunique() == 4
        folds = {
            fold
            for fold, frame in result.folds.items()
            if (frame.patient_id == "1002").any()
        }
        assert len(folds) == 1


class TestEchoNextSplitter:
    """EchoNext: predefined splits over rows of shared arrays, minus no_split."""

    def _metadata(self, tmp_path):
        """A stand-in for echonext_metadata_100k.csv, with the leakage hazard in it."""
        from ecgbench.splitting.strategies.echonext import SOURCE_METADATA

        rows = [
            # patient, split -- p1's earlier ECG is no_split, its latest is test,
            # which is exactly the pair that would leak if no_split became train.
            ("e1", "p1", "no_split", 0),
            ("e2", "p1", "test", 1),
            ("e3", "p2", "train", 1),
            ("e4", "p2", "train", 0),
            ("e5", "p3", "val", 1),
            ("e6", "p4", "no_split", 0),
            ("e7", "p4", "val", 0),
        ]
        df = pd.DataFrame(rows, columns=["ecg_key", "patient_key", "split",
                                         "shd_moderate_or_greater_flag"])
        df.insert(0, "Unnamed: 0", range(len(df)))
        df.to_csv(tmp_path / SOURCE_METADATA, index=False)
        return tmp_path

    def _splitter(self):
        from ecgbench.splitting.strategies.echonext import EchoNextSplitter

        return EchoNextSplitter()

    def test_no_split_records_are_excluded(self, tmp_path):
        """Folding them into train would put test patients' ECGs in training."""
        df = self._splitter().load_metadata(self._metadata(tmp_path), None)

        assert len(df) == 5
        assert "no_split" not in set(df["split"])
        assert set(df["ecg_key"]) == {"e2", "e3", "e4", "e5", "e7"}

    def test_the_remaining_splits_are_patient_disjoint(self, tmp_path):
        """The property excluding no_split buys, and the reason for excluding it."""
        df = self._splitter().load_metadata(self._metadata(tmp_path), None)

        by_split = df.groupby("split")["patient_key"].agg(set)
        for a, b in itertools.combinations(by_split.index, 2):
            assert not (by_split[a] & by_split[b]), f"{a} and {b} share a patient"

    def test_signal_path_carries_the_row_within_its_split(self, tmp_path):
        """Row index is into that split's array, so it must be taken before filtering."""
        df = self._splitter().load_metadata(self._metadata(tmp_path), None)
        paths = dict(zip(df["ecg_key"], df["signal_path"]))

        # e2 is p1's test ECG and the FIRST test row -> row 0 of the test array.
        assert paths["e2"] == "EchoNext_test_waveforms.npy:0"
        # e3/e4 are train rows 0 and 1.
        assert paths["e3"] == "EchoNext_train_waveforms.npy:0"
        assert paths["e4"] == "EchoNext_train_waveforms.npy:1"
        # e7 is the SECOND val record in file order (e5 is first), and keeps row 1
        # even though the no_split row between them was dropped.
        assert paths["e5"] == "EchoNext_val_waveforms.npy:0"
        assert paths["e7"] == "EchoNext_val_waveforms.npy:1"

    def test_row_index_is_not_renumbered_by_the_exclusion(self, tmp_path):
        """The array still contains the no_split rows; only our table drops them."""
        from ecgbench.splitting.strategies.echonext import SOURCE_METADATA

        # Two no_split rows ahead of a train row: the train row is still train row 0
        # because rows are numbered within their own split, not globally.
        pd.DataFrame({
            "ecg_key": ["a", "b", "c"],
            "patient_key": ["p1", "p2", "p3"],
            "split": ["no_split", "no_split", "train"],
            "shd_moderate_or_greater_flag": [0, 1, 1],
        }).to_csv(tmp_path / SOURCE_METADATA, index=False)

        df = self._splitter().load_metadata(tmp_path, None)

        assert df["signal_path"].tolist() == ["EchoNext_train_waveforms.npy:0"]

    def test_folds_match_the_configs_mapping(self, tmp_path):
        from ecgbench.splitting.strategies.echonext import SPLIT_TO_FOLD

        df = self._splitter().load_metadata(self._metadata(tmp_path), None)

        assert SPLIT_TO_FOLD == {"train": 1, "val": 2, "test": 3}
        assert dict(zip(df["split"], df["fold"])) == {"train": 1, "val": 2, "test": 3}
        assert 4 not in set(df["fold"])          # no_split has no fold, by design

    def test_normalised_table_is_written_for_the_validator(self, tmp_path):
        """validate_dataset re-reads metadata_csv from disk and rebuilds paths."""
        from ecgbench.splitting.strategies.echonext import GENERATED_METADATA

        self._splitter().load_metadata(self._metadata(tmp_path), None)

        written = pd.read_csv(tmp_path / GENERATED_METADATA)
        assert "signal_path" in written.columns
        assert len(written) == 5
        assert written["signal_path"].str.contains(r"\.npy:\d+$").all()

    def test_stratification_is_the_composite_shd_label(self, tmp_path):
        splitter = self._splitter()
        df = splitter.load_metadata(self._metadata(tmp_path), None)

        labels = splitter.get_stratification_labels(df, None)

        assert labels.tolist() == [1, 1, 0, 1, 0]

    def test_missing_source_names_the_release(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="physionet.org/content/echonext"):
            self._splitter().load_metadata(tmp_path, None)


class TestSTAFFIIISplitter:
    """STAFF III's binding constraints: patient grouping and an .xlsx source.

    The metadata cache is not an optimisation here — ``validate_dataset``
    re-reads ``metadata_csv`` with ``pandas.read_csv``, which cannot open the
    spreadsheet the labels actually live in. If ``load_metadata`` did not write
    the normalised CSV to disk, validation would have no metadata at all.
    """

    N_COLUMNS = 29

    def _header(self, rec, n_samples=300000):
        return (
            f"{rec} 9 1000 {n_samples} 20:26:00 27/09/1995\n"
            + "".join(
                f"{rec}.dat 16+512 1600 12 0 0 0 0  {lead}\n"
                for lead in ("V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III")
            )
            + "# Age: 52\n# Sex: F\n"
        )

    def _sheet_row(self, patient, **cells):
        row = [None] * self.N_COLUMNS
        row[0], row[1], row[2], row[28] = patient, 52, "f", "no"
        for column, value in cells.items():
            row[int(column.lstrip("c"))] = value
        return row

    def _tree(self, tmp_path, rows, records):
        """Build a version directory: the spreadsheet plus data/NNNx.hea files."""
        pytest.importorskip("openpyxl")
        import pandas as pd

        preamble = [[None] * self.N_COLUMNS for _ in range(10)]
        pd.DataFrame(preamble + rows).to_excel(
            tmp_path / "STAFF-III-Database-Annotations.xlsx",
            header=False,
            index=False,
        )
        (tmp_path / "data").mkdir(exist_ok=True)
        for rec in records:
            (tmp_path / "data" / f"{rec}.hea").write_text(
                self._header(rec), encoding="utf-8"
            )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="staffiii", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="patient_id",
            signal_path_columns={1000: "signal_path"}, default_sampling_rate=1000,
            label_column="recording_type", leads=9,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.staffiii import STAFFIIISplitter

        assert isinstance(get_splitter("staffiii"), STAFFIIISplitter)

    def test_builds_patient_id_signal_path_and_phase(self, sample_config, tmp_path):
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [self._sheet_row(1, c3="1a", c4="1b", c6="1c", c7="mid RCA", c26="1d")],
            ["001a", "001b", "001c", "001d"],
        )
        df = get_splitter("staffiii").load_metadata(tree, config)

        assert len(df) == 4
        # All four recordings are one patient — that is what grouping depends on.
        assert set(df["patient_id"]) == {"patient001"}
        # Signals live one level down, so the path carries the data/ prefix.
        assert sorted(df["signal_path"]) == [
            "data/001a", "data/001b", "data/001c", "data/001d"
        ]
        phases = dict(zip(df["record_name"], df["recording_type"]))
        assert phases == {"001a": "BR", "001b": "BC", "001c": "BI", "001d": "PR"}
        assert phases["001c"] == "BI"
        # Written to disk because validate_dataset re-reads it — and cannot read
        # the .xlsx the labels came from.
        assert (tree / config.metadata_csv).exists()

    def test_multi_inflation_records_collapse_to_one_row(self, sample_config, tmp_path):
        """Nine real records hold 2-3 inflations; the frame stays one row each."""
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [
                self._sheet_row(
                    7, c3="7a", c6="7c", c7="dist circ", c10="7c", c11="prox circ"
                )
            ],
            ["007a", "007c"],
        )
        df = get_splitter("staffiii").load_metadata(tree, config)

        assert len(df) == 2  # not 3, even though the sheet lists 7c twice
        inflation = df[df["record_name"] == "007c"].iloc[0]
        assert inflation["occluded_artery"] == "dist circ;prox circ"
        assert inflation["artery_territory"] == "LCx;LCx"

    def test_cached_csv_keeps_zero_padded_ids_as_strings(self, sample_config, tmp_path):
        """Re-reading must not turn 'patient001' into a float or '001a' into 1."""
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [self._sheet_row(1, c3="1a", c6="1c", c7="mid RCA")],
            ["001a", "001c"],
        )
        splitter = get_splitter("staffiii")
        splitter.load_metadata(tree, config)  # writes the cache
        cached = splitter.load_metadata(tree, config)  # reads it back

        assert cached["record_name"].tolist() == ["001a", "001c"]
        assert cached["patient_id"].tolist() == ["patient001", "patient001"]
        assert cached["signal_path"].tolist() == ["data/001a", "data/001c"]

    def test_stratifies_on_territory_not_on_the_protocol_phase(
        self, sample_config, tmp_path
    ):
        """Folds are patient-grouped, so only patient-level attributes balance.

        Every patient contributes roughly the same mix of phases, so
        recording_type is near-uniform within a patient and cannot be stratified
        on; the occluded vessel is what actually varies between patients.
        """
        config = self._config(sample_config)
        rows, records = [], []
        for patient in range(1, 21):
            artery = "prox LAD" if patient <= 10 else "prox RCA"
            rows.append(
                self._sheet_row(
                    patient,
                    **{"c3": f"{patient}a", "c6": f"{patient}c", "c7": artery},
                )
            )
            records += [f"{patient:03d}a", f"{patient:03d}c"]
        tree = self._tree(tmp_path, rows, records)

        splitter = get_splitter("staffiii")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "artery_territory"
        assert set(labels) == {"LAD", "RCA"}
        # Both of a patient's records carry the patient's territory, including
        # the baseline — that is what keeps the patient in one fold coherent.
        assert labels[df["record_name"] == "001a"].tolist() == ["LAD"]

    def test_stratification_needs_the_label_loaders_column(self, sample_config):
        """A raw frame must fail loudly rather than stratify on something else."""
        import pandas as pd

        config = self._config(sample_config)
        df = pd.DataFrame({"record_name": ["001a"], "recording_type": ["BR"]})

        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("staffiii").get_stratification_labels(df, config)


class TestCPSC2018Splitter:
    """CPSC 2018 ships a flat record tree and no metadata file of any kind."""

    def _header(self, name, dx, nsamp=5000, age="54", sex="Female"):
        """The Kaggle mirror's header dialect: '16+24', '1000/mV', no space after '#'."""
        return (
            f"{name} 12 500 {nsamp} 12-May-2020 12:33:59\n"
            + "".join(
                f"{name}.mat 16+24 1000/mV 16 0 0 0 0 {lead}\n"
                for lead in ("I", "II", "III", "aVR", "aVL", "aVF",
                             "V1", "V2", "V3", "V4", "V5", "V6")
            )
            + f"#Age: {age}\n#Sex: {sex}\n#Dx: {dx}\n"
            "#Rx: Unknown\n#Hx: Unknown\n#Sx: Unknown\n"
        )

    def _tree(self, tmp_path, records, nsamp=5000):
        d = tmp_path / "Training_WFDB"
        d.mkdir(parents=True, exist_ok=True)
        for name, dx in records:
            (d / f"{name}.hea").write_text(
                self._header(name, dx, nsamp=nsamp), encoding="utf-8"
            )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="cpsc_2018", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={500: "signal_path"}, default_sampling_rate=500,
            label_column="dx", leads=12,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.cpsc_2018 import CPSC2018Splitter

        assert isinstance(get_splitter("cpsc_2018"), CPSC2018Splitter)

    def test_builds_signal_paths_from_a_flat_tree_and_caches_csv(
        self, sample_config, tmp_path
    ):
        """No cohort subdirectories here, unlike the challenge releases."""
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("A0001", "59118001"), ("A0002", "426783006")])
        df = get_splitter("cpsc_2018").load_metadata(tree, config)

        paths = dict(zip(df["record_name"], df["signal_path"]))
        assert paths["A0001"] == "Training_WFDB/A0001.mat"
        assert paths["A0002"] == "Training_WFDB/A0002.mat"
        # Written to disk because validate_dataset re-reads it rather than
        # reusing this frame.
        assert (tree / config.metadata_csv).exists()

    def test_metadata_exposes_all_nine_classes_and_the_duration(
        self, sample_config, tmp_path
    ):
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path, [("A0043", "164889003,59118001")], nsamp=72000
        )
        row = get_splitter("cpsc_2018").load_metadata(tree, config).iloc[0]

        assert row["dx"] == "164889003,59118001"
        assert row["n_dx"] == 2
        assert row["dx_abbreviations"] == "AF,RBBB"
        # Length varies by a factor of 24, so it is exposed per record.
        assert row["duration_seconds"] == 144.0

    def test_stratification_takes_the_rarest_class_not_the_first(
        self, sample_config, tmp_path
    ):
        """The shipped #Dx order is a class-index sort, so it must not be used."""
        config = self._config(sample_config)
        tree = self._tree(tmp_path, (
            [(f"A{i:04d}", "164889003") for i in range(12)]
            + [("A9999", "164889003,164931005")]
        ))
        df = get_splitter("cpsc_2018").load_metadata(tree, config)

        strat = dict(zip(df["record_name"], df["stratify_dx_abbreviation"]))
        assert strat["A9999"] == "STE"     # rarest, though AF is listed first
        assert strat["A0000"] == "AF"

    def test_rare_classes_are_not_pooled(self, sample_config, tmp_path):
        """Unlike Challenge 2020: the smallest real class here has 220 records.

        Pooling would silently rename a genuine class, so the splitter must
        leave every class it is given intact.
        """
        config = self._config(sample_config)
        tree = self._tree(tmp_path, (
            [(f"A{i:04d}", "164889003") for i in range(12)]
            + [("A9999", "164931005")]
        ))
        splitter = get_splitter("cpsc_2018")
        counts = splitter.get_stratification_labels(
            splitter.load_metadata(tree, config), config
        ).value_counts().to_dict()

        assert counts == {"AF": 12, "STE": 1}
        assert "OTHER" not in counts

    def test_stratification_needs_load_metadata_first(self, sample_config):
        import pandas as pd

        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("cpsc_2018").get_stratification_labels(
                pd.DataFrame({"record_name": ["A0001"]}), self._config(sample_config)
            )

    def test_missing_record_tree_raises(self, sample_config, tmp_path):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="Training_WFDB"):
            get_splitter("cpsc_2018").load_metadata(
                tmp_path, self._config(sample_config)
            )


class TestSPHSplitter:
    """SPH ships metadata.csv with no path column and codes needing a join."""

    CODE_CSV = (
        "Category,Code,Description\n"
        "A,1,Normal ECG\n"
        "C,22,Sinus bradycardia\n"
        "F,60,Ventricular premature complex(es)\n"
        "M,166,Extensive anterior MI\n"
        "Modifier,310,Frequent\n"
    )

    def _tree(self, tmp_path, rows):
        """rows: list of (ecg_id, aha_code, patient_id)."""
        (tmp_path / "code.csv").write_text(self.CODE_CSV, encoding="utf-8")
        lines = ["ECG_ID,AHA_Code,Patient_ID,Age,Sex,N,Date"]
        lines += [f"{rid},{code},{pid},44,M,5000,2020-01-01" for rid, code, pid in rows]
        (tmp_path / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="sph", metadata_csv="ecgbench_metadata.csv",
            record_id_column="ecg_id", patient_id_column="patient_id",
            signal_path_columns={500: "signal_path"}, default_sampling_rate=500,
            label_column="aha_primary_codes", leads=12, signal_format="hdf5",
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.sph import SPHSplitter

        assert isinstance(get_splitter("sph"), SPHSplitter)

    def test_builds_the_implicit_signal_path_and_caches_the_csv(self, tmp_path, sample_config):
        """The release leaves records/<ECG_ID>.h5 to convention; validation cannot."""
        config = self._config(sample_config)
        root = self._tree(tmp_path, [("A00001", "1", "S1"), ("A00002", "22", "S2")])

        df = get_splitter("sph").load_metadata(root, config)

        assert list(df["signal_path"]) == ["records/A00001.h5", "records/A00002.h5"]
        assert list(df["ecg_id"]) == ["A00001", "A00002"]
        # Written to disk, because validate_dataset re-reads it rather than
        # reusing this frame.
        cached = root / "ecgbench_metadata.csv"
        assert cached.exists()
        again = get_splitter("sph").load_metadata(root, config)
        assert list(again["signal_path"]) == list(df["signal_path"])
        # Codes must survive the CSV round trip as strings, not become ints.
        assert again["aha_primary_codes"].map(type).eq(str).all()

    def test_stratifies_on_the_rarest_code_and_pools_the_tail(self, tmp_path, sample_config):
        """9 of the 44 real codes fall under ten records; they must not each be a class."""
        from ecgbench.splitting.strategies.sph import MIN_CLASS_SIZE, OTHER

        assert MIN_CLASS_SIZE == 10
        config = self._config(sample_config)
        rows = [(f"A{i:05d}", "1", f"S{i}") for i in range(12)]
        rows += [(f"B{i:05d}", "22", f"T{i}") for i in range(11)]
        rows.append(("C00001", "166", "U1"))  # a single rare record
        splitter = get_splitter("sph")
        labels = splitter.get_stratification_labels(
            splitter.load_metadata(self._tree(tmp_path, rows), config), config
        )

        counts = labels.value_counts().to_dict()
        assert counts == {"1": 12, "22": 11, OTHER: 1}
        assert "166" not in counts

    def test_a_multi_label_record_strata_on_its_rarest_code(self, tmp_path, sample_config):
        """Statement order is not a ranking, so the first listed code is not used."""
        config = self._config(sample_config)
        rows = [(f"A{i:05d}", "22", f"S{i}") for i in range(15)]
        rows += [(f"B{i:05d}", "60", f"T{i}") for i in range(10)]
        rows.append(("C00001", "22;60", "U1"))
        splitter = get_splitter("sph")
        df = splitter.load_metadata(self._tree(tmp_path, rows), config)
        labels = splitter.get_stratification_labels(df, config)

        # 22 occurs 16 times, 60 eleven, so the multi-label record goes to 60.
        assert df.set_index("ecg_id").loc["C00001", "aha_primary_codes"] == "22;60"
        assert labels[df.index[df["ecg_id"] == "C00001"][0]] == "60"

    def test_no_class_is_pooled_when_every_class_is_big_enough(self, tmp_path, sample_config):
        from ecgbench.splitting.strategies.sph import OTHER

        config = self._config(sample_config)
        rows = [(f"A{i:05d}", "1", f"S{i}") for i in range(12)]
        rows += [(f"B{i:05d}", "22", f"T{i}") for i in range(10)]
        splitter = get_splitter("sph")
        counts = splitter.get_stratification_labels(
            splitter.load_metadata(self._tree(tmp_path, rows), config), config
        ).value_counts().to_dict()

        assert counts == {"1": 12, "22": 10}
        assert OTHER not in counts

    def test_stratification_needs_load_metadata_first(self, sample_config):
        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("sph").get_stratification_labels(
                pd.DataFrame({"ecg_id": ["A00001"]}), self._config(sample_config)
            )

    def test_missing_metadata_raises_naming_the_file(self, sample_config, tmp_path):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="metadata.csv"):
            get_splitter("sph").load_metadata(tmp_path, self._config(sample_config))

    def test_folds_keep_a_repeated_patient_together(self, tmp_path, sample_config):
        """1,066 real patients have 2-5 records, so grouping has to actually hold."""
        config = self._config(sample_config)
        # 40 patients, every fourth contributing two records.
        rows = []
        for i in range(40):
            code = "1" if i % 2 else "22"
            rows.append((f"A{i:05d}", code, f"S{i}"))
            if i % 4 == 0:
                rows.append((f"B{i:05d}", code, f"S{i}"))
        splitter = get_splitter("sph")
        df = splitter.load_metadata(self._tree(tmp_path, rows), config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)
        assigned = pd.concat(
            [frame.assign(fold=n) for n, frame in result.folds.items()], ignore_index=True
        )
        assert assigned.groupby("patient_id")["fold"].nunique().max() == 1
        assert result.group_column == "patient_id"


class TestNingboIVASplitter:
    """Ningbo IVA ships a spreadsheet with five columns and none of them a path."""

    def _sheet(self, tmp_path, rows):
        """rows: list of (hospital_id, type, left_right, sublocation, gender)."""
        pd.DataFrame(
            rows, columns=["HospitalID", "Type", "LeftRight", "Sublocation", "Gender"]
        ).to_csv(tmp_path / "Diagnosis.csv", index=False)
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ningbo_iva", metadata_csv="ecgbench_metadata.csv",
            record_id_column="hospital_id", patient_id_column=None,
            signal_path_columns={2000: "signal_path"}, default_sampling_rate=2000,
            label_column="left_right", leads=12, signal_format="csv",
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.ningbo_iva import NingboIVASplitter

        assert isinstance(get_splitter("ningbo_iva"), NingboIVASplitter)

    def test_builds_the_missing_signal_path_and_caches_the_csv(self, tmp_path, sample_config):
        """A path fix-up that lives only in memory leaves validation with nothing."""
        config = self._config(sample_config)
        root = self._sheet(
            tmp_path,
            [
                (1000364, "PVC", "Right", "AC", "female"),
                (991591, "VT", "Left", "LCC", "male"),
            ],
        )

        df = get_splitter("ningbo_iva").load_metadata(root, config)

        assert set(df["signal_path"]) == {
            "PVCVTRawECGData/1000364.csv",
            "PVCVTRawECGData/991591.csv",
        }
        cached = root / "ecgbench_metadata.csv"
        assert cached.exists()
        # The generated file is what validate_dataset reads, so the column has to
        # be there on the second pass too — and the ids must stay strings.
        again = get_splitter("ningbo_iva").load_metadata(root, config)
        assert again["hospital_id"].map(type).eq(str).all()
        assert set(again["signal_path"]) == set(df["signal_path"])

    def test_stratifies_on_the_ablation_confirmed_tract(self, tmp_path, sample_config):
        config = self._config(sample_config)
        rows = [(i, "PVC", "Right", "AC", "female") for i in range(1, 8)]
        rows += [(i, "PVC", "Left", "LCC", "male") for i in range(100, 103)]
        splitter = get_splitter("ningbo_iva")
        labels = splitter.get_stratification_labels(
            splitter.load_metadata(self._sheet(tmp_path, rows), config), config
        )

        assert labels.value_counts().to_dict() == {"RVOT": 7, "LVOT": 3}
        assert labels.name == "stratify_class"

    def test_sublocation_is_not_the_split_target(self, tmp_path, sample_config):
        """12 values over 334 patients, five under ten cases — unusable for 10 folds."""
        from ecgbench.splitting.strategies.ningbo_iva import STRATIFY_COLUMN

        assert STRATIFY_COLUMN == "left_right"
        config = self._config(sample_config)
        rows = [(i, "PVC", "Right", f"Site{i % 6}", "female") for i in range(1, 13)]
        splitter = get_splitter("ningbo_iva")
        labels = splitter.get_stratification_labels(
            splitter.load_metadata(self._sheet(tmp_path, rows), config), config
        )

        assert set(labels) == {"RVOT"}

    def test_stratification_needs_load_metadata_first(self, sample_config):
        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("ningbo_iva").get_stratification_labels(
                pd.DataFrame({"hospital_id": ["1000364"]}), self._config(sample_config)
            )

    def test_missing_spreadsheet_raises_naming_the_file(self, sample_config, tmp_path):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="Diagnosis"):
            get_splitter("ningbo_iva").load_metadata(tmp_path, self._config(sample_config))


class TestCODE15Splitter:
    """CODE-15% resolves exam_id to a row of one of 18 shared HDF5 arrays."""

    COLUMNS = (
        "exam_id,age,is_male,nn_predicted_age,1dAVb,RBBB,LBBB,SB,ST,AF,"
        "patient_id,death,timey,normal_ecg,trace_file"
    )

    def _tree(self, tmp_path, rows, hdf5_ids=None):
        """rows: (exam_id, patient_id, trace_file, rbbb). hdf5_ids: part -> ids."""
        h5py = pytest.importorskip("h5py")
        lines = [self.COLUMNS]
        for exam_id, patient_id, part, rbbb in rows:
            lines.append(
                f"{exam_id},50,True,51.0,False,{rbbb},False,False,False,False,"
                f"{patient_id},False,1.0,{not rbbb},{part}"
            )
        (tmp_path / "exams.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        if hdf5_ids is None:
            # Default: each part's own order, plus the trailing all-zero padding
            # row with exam_id 0 that every real part carries.
            hdf5_ids = {}
            for exam_id, _, part, _ in rows:
                hdf5_ids.setdefault(part, []).append(exam_id)
            hdf5_ids = {p: [*ids, 0] for p, ids in hdf5_ids.items()}

        for part, ids in hdf5_ids.items():
            with h5py.File(tmp_path / part, "w") as handle:
                handle.create_dataset("exam_id", data=np.array(ids, dtype=np.int64))
                handle.create_dataset(
                    "tracings", data=np.zeros((len(ids), 8, 12), dtype=np.float32)
                )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="code15", metadata_csv="ecgbench_metadata.csv",
            record_id_column="exam_id", patient_id_column="patient_id",
            signal_path_columns={400: "signal_path"}, default_sampling_rate=400,
            label_column="abnormality_codes", leads=12, signal_format="hdf5",
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.code15 import CODE15Splitter

        assert isinstance(get_splitter("code15"), CODE15Splitter)

    def test_row_index_comes_from_the_hdf5_not_from_the_csv_order(
        self, tmp_path, sample_config
    ):
        """exams.csv is not in file order — part 0's real CSV rows and its own
        exam_id dataset disagree, so a positional read mislabels almost every
        record."""
        config = self._config(sample_config)
        rows = [
            (500, 1, "exams_part0.hdf5", False),
            (600, 2, "exams_part0.hdf5", True),
            (700, 3, "exams_part0.hdf5", False),
        ]
        # The file stores them in a different order from the CSV.
        root = self._tree(tmp_path, rows, hdf5_ids={"exams_part0.hdf5": [700, 500, 600, 0]})

        df = get_splitter("code15").load_metadata(root, config).set_index("exam_id")

        assert df.loc[700, "signal_path"] == "exams_part0.hdf5:tracings:0"
        assert df.loc[500, "signal_path"] == "exams_part0.hdf5:tracings:1"
        assert df.loc[600, "signal_path"] == "exams_part0.hdf5:tracings:2"

    def test_the_trailing_padding_row_is_dropped(self, tmp_path, sample_config):
        """Every part holds one more row than it has records: an all-zero row
        with exam_id 0 that appears in no CSV."""
        config = self._config(sample_config)
        root = self._tree(tmp_path, [
            (500, 1, "exams_part0.hdf5", False),
            (600, 2, "exams_part0.hdf5", False),
        ])

        df = get_splitter("code15").load_metadata(root, config)

        assert len(df) == 2
        assert 0 not in set(df["exam_id"])
        assert not any(p.endswith(":2") for p in df["signal_path"])

    def test_records_spread_across_parts_each_get_their_own_file(
        self, tmp_path, sample_config
    ):
        config = self._config(sample_config)
        root = self._tree(tmp_path, [
            (500, 1, "exams_part0.hdf5", False),
            (600, 2, "exams_part9.hdf5", True),
        ])

        df = get_splitter("code15").load_metadata(root, config).set_index("exam_id")

        assert df.loc[500, "signal_path"] == "exams_part0.hdf5:tracings:0"
        assert df.loc[600, "signal_path"] == "exams_part9.hdf5:tracings:0"

    def test_a_part_disagreeing_with_the_csv_raises(self, tmp_path, sample_config):
        """Zenodo publishes checksums only for the zips, so this set comparison
        is the only integrity check an extracted copy gets."""
        config = self._config(sample_config)
        root = self._tree(
            tmp_path,
            [(500, 1, "exams_part0.hdf5", False), (600, 2, "exams_part0.hdf5", False)],
            hdf5_ids={"exams_part0.hdf5": [500, 999, 0]},   # 600 missing, 999 extra
        )

        with pytest.raises(ValueError, match="disagree about which records"):
            get_splitter("code15").load_metadata(root, config)

    def test_a_missing_part_names_the_file(self, tmp_path, sample_config):
        config = self._config(sample_config)
        root = self._tree(tmp_path, [(500, 1, "exams_part0.hdf5", False)])
        (root / "exams_part0.hdf5").unlink()

        with pytest.raises(FileNotFoundError, match="exams_part0.hdf5"):
            get_splitter("code15").load_metadata(root, config)

    def test_metadata_is_written_to_disk_for_the_validation_engine(
        self, tmp_path, sample_config
    ):
        """validate_dataset re-reads config.metadata_csv and rebuilds paths from
        it, so an in-memory-only fix-up fails every record."""
        config = self._config(sample_config)
        root = self._tree(tmp_path, [(500, 1, "exams_part0.hdf5", False)])

        get_splitter("code15").load_metadata(root, config)

        written = root / "ecgbench_metadata.csv"
        assert written.exists()
        reread = pd.read_csv(written)
        assert reread.loc[0, "signal_path"] == "exams_part0.hdf5:tracings:0"

    def test_cached_metadata_keeps_the_empty_label_list_a_string(
        self, tmp_path, sample_config
    ):
        """abnormality_codes is empty for 89% of records; read back naively it
        becomes float NaN and stops being a list."""
        config = self._config(sample_config)
        root = self._tree(tmp_path, [
            (500, 1, "exams_part0.hdf5", False), (600, 2, "exams_part0.hdf5", True),
        ])
        splitter = get_splitter("code15")

        first = splitter.load_metadata(root, config)
        cached = splitter.load_metadata(root, config)      # now hits the cache

        assert list(cached["abnormality_codes"]) == list(first["abnormality_codes"])
        assert cached["abnormality_codes"].iloc[0] == ""
        assert cached["abnormality_codes"].iloc[1] == "RBBB"

    def test_stratification_separates_normal_from_other(self, tmp_path, sample_config):
        config = self._config(sample_config)
        root = self._tree(tmp_path, [
            (500, 1, "exams_part0.hdf5", False),    # normal_ecg True
            (600, 2, "exams_part0.hdf5", True),     # RBBB
        ])
        splitter = get_splitter("code15")
        df = splitter.load_metadata(root, config)

        labels = splitter.get_stratification_labels(df, config)
        assert labels.name == "stratify_class"
        assert list(labels) == ["NORMAL", "RBBB"]

    def test_stratification_without_load_metadata_first_says_so(self, sample_config):
        splitter = get_splitter("code15")
        with pytest.raises(ValueError, match="call load_metadata"):
            splitter.get_stratification_labels(
                pd.DataFrame({"exam_id": [1]}), self._config(sample_config)
            )


class TestCODETestSplitter:
    """CODE-test has no identifiers: the record id is the array row index."""

    ABN = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")

    def _tree(self, tmp_path, n=4, flags=None, n_array_rows=None):
        h5py = pytest.importorskip("h5py")
        (tmp_path / "annotations").mkdir(exist_ok=True)
        (tmp_path / "attributes.csv").write_text(
            "\n".join(["age,sex"] + [f"{30 + i},M" for i in range(n)]) + "\n",
            encoding="utf-8",
        )

        from ecgbench.labels.code_test import ANNOTATORS

        for name in ANNOTATORS:
            rows = flags if (flags and name == "gold_standard") else [()] * n
            header = ("," if name == "dnn" else "") + ",".join(self.ABN)
            lines = [header]
            for i, on in enumerate(rows):
                cells = ["1" if c in on else "0" for c in self.ABN]
                lines.append(",".join(([str(i)] if name == "dnn" else []) + cells))
            (tmp_path / "annotations" / f"{name}.csv").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

        with h5py.File(tmp_path / "ecg_tracings.hdf5", "w") as handle:
            handle.create_dataset(
                "tracings",
                data=np.zeros((n_array_rows if n_array_rows else n, 8, 12)),
            )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="code_test", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_id", patient_id_column=None,
            signal_path_columns={400: "signal_path"}, default_sampling_rate=400,
            label_column="abnormality_codes", leads=12, signal_format="hdf5",
        )

    def _patch_n(self, monkeypatch, n):
        import ecgbench.labels.code_test as labels_mod
        import ecgbench.splitting.strategies.code_test as split_mod

        monkeypatch.setattr(labels_mod, "N_RECORDS", n)
        monkeypatch.setattr(split_mod, "N_RECORDS", n)

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.code_test import CODETestSplitter

        assert isinstance(get_splitter("code_test"), CODETestSplitter)

    def test_signal_paths_are_row_references_into_the_shared_array(
        self, tmp_path, sample_config, monkeypatch
    ):
        self._patch_n(monkeypatch, 3)
        config = self._config(sample_config)
        df = get_splitter("code_test").load_metadata(self._tree(tmp_path, 3), config)

        assert list(df["record_id"]) == [0, 1, 2]
        assert list(df["signal_path"]) == [
            "ecg_tracings.hdf5:tracings:0",
            "ecg_tracings.hdf5:tracings:1",
            "ecg_tracings.hdf5:tracings:2",
        ]

    def test_an_array_of_the_wrong_length_is_refused(
        self, tmp_path, sample_config, monkeypatch
    ):
        """Record ids are row numbers, so a different row count means the
        alignment the whole release rests on no longer holds."""
        self._patch_n(monkeypatch, 3)
        config = self._config(sample_config)
        root = self._tree(tmp_path, 3, n_array_rows=5)

        with pytest.raises(ValueError, match="aligned to that array by row position"):
            get_splitter("code_test").load_metadata(root, config)

    def test_missing_tracings_file_points_at_the_data_subdirectory(
        self, tmp_path, sample_config, monkeypatch
    ):
        self._patch_n(monkeypatch, 2)
        config = self._config(sample_config)
        root = self._tree(tmp_path, 2)
        (root / "ecg_tracings.hdf5").unlink()

        with pytest.raises(FileNotFoundError, match="data/"):
            get_splitter("code_test").load_metadata(root, config)

    def test_metadata_is_written_to_disk_and_reread_intact(
        self, tmp_path, sample_config, monkeypatch
    ):
        self._patch_n(monkeypatch, 3)
        config = self._config(sample_config)
        root = self._tree(tmp_path, 3, flags=[("AF",), (), ("RBBB",)])
        splitter = get_splitter("code_test")

        first = splitter.load_metadata(root, config)
        assert (root / "ecgbench_metadata.csv").exists()

        cached = splitter.load_metadata(root, config)
        assert list(cached["abnormality_codes"]) == list(first["abnormality_codes"])
        # Empty for the unflagged record, and still a string rather than NaN.
        assert cached["abnormality_codes"].iloc[1] == ""
        assert cached["gold_standard_abnormality_codes"].iloc[1] == ""

    def test_no_patient_grouping_because_no_patient_id_ships(self):
        """827 tracings from 827 different patients, and no identifier at all."""
        from ecgbench.config import load_config

        assert load_config("code_test").patient_id_column is None

    def test_stratification_labels_come_from_the_gold_standard(
        self, tmp_path, sample_config, monkeypatch
    ):
        self._patch_n(monkeypatch, 3)
        config = self._config(sample_config)
        root = self._tree(tmp_path, 3, flags=[("AF", "RBBB"), (), ("RBBB",)])
        splitter = get_splitter("code_test")
        df = splitter.load_metadata(root, config)

        labels = splitter.get_stratification_labels(df, config)
        assert labels.name == "stratify_class"
        assert list(labels) == ["AF", "NONE", "RBBB"]


class TestSamiTropSplitter:
    """SaMi-Trop: a row reference into one array, with no key to check it against."""

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="sami_trop", signal_format="hdf5",
            metadata_csv="ecgbench_metadata.csv", record_id_column="exam_id",
            patient_id_column=None, signal_path_columns={400: "signal_path"},
            default_sampling_rate=400, sampling_rates=[400],
        )

    def _hdf5(self, tmp_path, n_records=3, n_samples=4096, n_leads=12):
        h5py = pytest.importorskip("h5py")

        with h5py.File(tmp_path / "exams.hdf5", "w") as handle:
            handle.create_dataset(
                "tracings", data=np.zeros((n_records, n_samples, n_leads), dtype=np.float32)
            )
        return tmp_path

    def test_registered_under_its_config_slug(self):
        from ecgbench.splitting.strategies.sami_trop import SamiTropSplitter

        assert isinstance(get_splitter("sami_trop"), SamiTropSplitter)

    def test_paths_name_a_row_of_the_single_array(self, tmp_path, sample_config):
        from ecgbench.splitting.strategies.sami_trop import build_signal_paths

        root = self._hdf5(tmp_path, n_records=3)
        rows = pd.Series([0, 1, 2], index=pd.Index([77, 11, 42], name="exam_id"))
        paths = build_signal_paths(root, rows, self._config(sample_config))

        assert list(paths) == [
            "exams.hdf5:tracings:0",
            "exams.hdf5:tracings:1",
            "exams.hdf5:tracings:2",
        ]
        # Keyed by exam_id in the CSV's order, not sorted — the row IS the join.
        assert list(paths.index) == [77, 11, 42]

    def test_a_row_count_mismatch_raises_because_the_join_is_positional(
        self, tmp_path, sample_config
    ):
        """The array is the only check available on a keyless positional join.

        CODE-15% can verify its mapping against a per-part exam_id dataset;
        SaMi-Trop has none, so if the array and the CSV disagree on length there
        is nothing to fall back on and every record would be mislabelled.
        """
        from ecgbench.splitting.strategies.sami_trop import build_signal_paths

        root = self._hdf5(tmp_path, n_records=2)
        rows = pd.Series([0, 1, 2], index=pd.Index([1, 2, 3], name="exam_id"))
        with pytest.raises(ValueError, match="row position|2 records"):
            build_signal_paths(root, rows, self._config(sample_config))

    def test_a_missing_archive_says_where_it_comes_from(self, tmp_path, sample_config):
        from ecgbench.splitting.strategies.sami_trop import build_signal_paths

        rows = pd.Series([0], index=pd.Index([1], name="exam_id"))
        with pytest.raises(FileNotFoundError, match="exams.zip"):
            build_signal_paths(tmp_path, rows, self._config(sample_config))


class TestIKEMSplitter:
    """IKEM: keyed across three parts, with shipped waveform hashes to lean on."""

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ikem", signal_format="hdf5",
            metadata_csv="ecgbench_metadata.csv", record_id_column="exam_id",
            patient_id_column="patient_id", signal_path_columns={500: "signal_path"},
            leads=8,
        )

    def _parts(self, tmp_path, parts):
        """parts: {filename: (exam_ids, hashes)}."""
        h5py = pytest.importorskip("h5py")

        for name, (exam_ids, hashes) in parts.items():
            with h5py.File(tmp_path / name, "w") as handle:
                handle.create_dataset(
                    "tracings", data=np.zeros((len(exam_ids), 4096, 8), dtype=np.int16)
                )
                handle.create_dataset("exam_id", data=np.array(exam_ids, dtype=np.int32))
                handle.create_dataset(
                    "hashes", data=np.array([h.encode() for h in hashes], dtype="S40")
                )
        return tmp_path

    def test_registered_under_its_config_slug(self):
        from ecgbench.splitting.strategies.ikem import IKEMSplitter

        assert isinstance(get_splitter("ikem"), IKEMSplitter)

    def test_rows_come_from_each_parts_own_exam_id_dataset(self, tmp_path, sample_config):
        """exams.csv has no trace_file column, so every part is scanned.

        Keying on exam_id rather than position also means the row order inside a
        part does not have to match anything — which is exactly where CODE-15%
        would break if it used positions.
        """
        from ecgbench.splitting.strategies.ikem import build_signal_paths

        root = self._parts(tmp_path, {
            "exams_part_1.hdf5": ([5, 1], ["a" * 40, "b" * 40]),
            "exams_part_2.hdf5": ([9], ["c" * 40]),
        })
        ids = pd.Index([1, 5, 9], name="exam_id")
        paths = build_signal_paths(root, ids, self._config(sample_config))

        # exam 5 is row 0 of part 1 and exam 1 is row 1, despite the id order.
        assert paths[5] == "exams_part_1.hdf5:tracings:0"
        assert paths[1] == "exams_part_1.hdf5:tracings:1"
        assert paths[9] == "exams_part_2.hdf5:tracings:0"

    def test_duplicate_waveform_hashes_mean_a_corrupted_copy(self, tmp_path, sample_config):
        """All 98,130 published hashes are distinct, so a repeat is not the data.

        This is a 1-D read that catches duplicated rows without touching any of
        the 6.6 GB of tracings.
        """
        from ecgbench.splitting.strategies.ikem import build_signal_paths

        root = self._parts(tmp_path, {
            "exams_part_1.hdf5": ([1, 2], ["a" * 40, "a" * 40]),
        })
        with pytest.raises(ValueError, match="same waveform SHA-1|duplicated rows"):
            build_signal_paths(root, pd.Index([1, 2], name="exam_id"), self._config(sample_config))

    def test_records_only_in_the_csv_are_reported_not_dropped(self, tmp_path, sample_config):
        from ecgbench.splitting.strategies.ikem import build_signal_paths

        root = self._parts(tmp_path, {"exams_part_1.hdf5": ([1], ["a" * 40])})
        with pytest.raises(ValueError, match="disagree about which records"):
            build_signal_paths(
                root, pd.Index([1, 2, 3], name="exam_id"), self._config(sample_config)
            )

    def test_an_exam_id_in_two_parts_raises(self, tmp_path, sample_config):
        """The parts are meant to be disjoint; overlap would make the row ambiguous."""
        from ecgbench.splitting.strategies.ikem import build_signal_paths

        root = self._parts(tmp_path, {
            "exams_part_1.hdf5": ([1], ["a" * 40]),
            "exams_part_2.hdf5": ([1], ["b" * 40]),
        })
        with pytest.raises(ValueError, match="more than one part"):
            build_signal_paths(root, pd.Index([1], name="exam_id"), self._config(sample_config))


class TestZZUPediatricSplitter:
    """ZZU-pECG: a metadata table that has to be normalised onto disk."""

    def test_registered_under_its_config_slug(self):
        from ecgbench.splitting.strategies.zzu_pecg import ZZUPediatricSplitter

        assert isinstance(get_splitter("zzu_pecg"), ZZUPediatricSplitter)

    def test_an_incomplete_split_zip_is_one_error_not_fourteen_thousand(
        self, tmp_path, sample_config
    ):
        """The waveforms ship as Child_ecg.zip + Child_ecg.z01 and must be joined.

        Unzipping only the .zip yields a partial tree. Without this check every
        absent record surfaces later as a separate ``corrupt_header`` failure,
        which reads as a data-quality problem rather than a missing download.
        """
        from dataclasses import replace

        from ecgbench.splitting.strategies.zzu_pecg import ZZUPediatricSplitter

        config = replace(sample_config, slug="zzu_pecg", record_id_column="ECG_ID",
                         patient_id_column="Patient_ID")
        df = pd.DataFrame({"signal_path": ["Child_ecg/P00/P00001/P00001_E01"]})
        missing = ZZUPediatricSplitter._missing_signals(tmp_path, df, config)
        assert missing == ["Child_ecg/P00/P00001/P00001_E01"]

        (tmp_path / "Child_ecg" / "P00" / "P00001").mkdir(parents=True)
        (tmp_path / "Child_ecg" / "P00" / "P00001" / "P00001_E01.hea").write_text("x")
        assert ZZUPediatricSplitter._missing_signals(tmp_path, df, config) == []

    def test_stratification_needs_the_disease_group_column(self, sample_config):
        """The ECG findings are multi-label with 99 codes, which cannot partition
        ten ways, so the folds use the ICD-10 disease group instead."""
        from dataclasses import replace

        from ecgbench.splitting.strategies.zzu_pecg import (
            STRATIFY_COLUMN,
            ZZUPediatricSplitter,
        )

        config = replace(sample_config, slug="zzu_pecg")
        splitter = ZZUPediatricSplitter()
        with pytest.raises(ValueError, match=STRATIFY_COLUMN):
            splitter.get_stratification_labels(pd.DataFrame({"ECG_ID": ["a"]}), config)

        labels = splitter.get_stratification_labels(
            pd.DataFrame({STRATIFY_COLUMN: ["NONE", "Kawasaki disease"]}), config
        )
        assert list(labels) == ["NONE", "Kawasaki disease"]


class TestMedalCareXLSplitter:
    """MedalCare-XL: metadata built from directory names, since none ships."""

    def _tree(self, tmp_path):
        """A miniature of the release's signal and parameter trees.

        Reproduces the four structural traps in the real one: the extra MI
        subclass level, the non-record ``mi/examples/`` directory, gaps in the
        per-directory record numbering, and three file variants per record.
        """
        from ecgbench.splitting.strategies.medalcare_xl import PARAMS_DIR, SIGNALS_DIR

        signals = tmp_path / SIGNALS_DIR
        params = tmp_path / PARAMS_DIR
        layout = {
            "sinus/train/run_S65": ["000001", "000002"],
            "sinus/test/run_S64": ["000001"],
            # A gap: 000002 is missing, exactly as iab/train/run_S66 has 23.
            "iab/validation/run_S67": ["000001", "000003"],
            "mi/LCX_0.3_ant/train/run_S69": ["000001"],
            "mi/RCA_1.0/test/run_S62": ["000001"],
        }
        for rel, numbers in layout.items():
            for number in numbers:
                (signals / rel).mkdir(parents=True, exist_ok=True)
                for variant in ("raw", "noise", "filtered"):
                    (signals / rel / f"{number}_{variant}.csv").write_text("0,0\n")
                (params / rel).mkdir(parents=True, exist_ok=True)
                for kind in ("Atrial", "Ventricular"):
                    (params / rel / f"{number}_{kind}Parameters.txt").write_text(
                        "im.name = Courtemanche\n"
                    )
            (signals / rel / "siginfo.csv").write_text("info1,info2\n")
        # Six loose illustration files with no split level — not records.
        (signals / "mi" / "examples").mkdir(parents=True)
        for name in ("S62_LAD_0.3", "S62_RCA_1.0"):
            for variant in ("raw", "noise", "filtered"):
                (signals / "mi" / "examples" / f"{name}_{variant}.csv").write_text("0,0\n")
        return tmp_path

    def _build(self, tmp_path):
        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.medalcare_xl import build_metadata

        return build_metadata(self._tree(tmp_path), load_config("medalcare_xl"))

    def test_splitter_is_registered(self):
        assert type(get_splitter("medalcare_xl")).__name__ == "MedalCareXLSplitter"

    def test_mi_examples_are_not_records(self, tmp_path):
        """``mi/examples/`` holds figure illustrations, not data.

        Counting ``*_raw.csv`` naively finds 6 more records than exist in the real
        release, and both parameter-file trees agree with the smaller number. The
        directory is excluded by having no train/validation/test level, so a future
        subclass directory that *does* have one still has to parse.
        """
        df = self._build(tmp_path)
        assert len(df) == 7
        assert not df["record_id"].str.contains("example").any()
        assert not df["signal_path"].str.contains("examples").any()

    def test_records_are_enumerated_from_the_files_not_a_range(self, tmp_path):
        """13 of the 186 real run directories have gaps in their numbering."""
        df = self._build(tmp_path)
        iab = df[df["pathology"] == "iab"]
        assert sorted(iab["record_number"]) == ["000001", "000003"]

    def test_record_ids_are_unique_across_the_whole_release(self, tmp_path):
        """Numbering restarts at 000001 per run directory, so the path is the key."""
        df = self._build(tmp_path)
        assert df["record_id"].is_unique
        # The same number appears in five directories and yields five distinct ids.
        assert (df["record_number"] == "000001").sum() == 5
        assert "sinus_train_S65_000001" in set(df["record_id"])
        assert "mi_LCX_0.3_ant_train_S69_000001" in set(df["record_id"])

    def test_signal_path_is_the_filtered_variant_and_the_others_are_kept(self, tmp_path):
        """One record in three renderings, not three records."""
        df = self._build(tmp_path)
        assert df["signal_path"].str.endswith("_filtered.csv").all()
        assert df["signal_path_raw"].str.endswith("_raw.csv").all()
        assert df["signal_path_noise"].str.endswith("_noise.csv").all()
        # Each record contributes exactly one row despite its three files.
        assert len(df) == df["signal_path"].nunique()

    def test_mi_subclass_is_decomposed_and_blank_elsewhere(self, tmp_path):
        df = self._build(tmp_path).set_index("record_id")
        lcx = df.loc["mi_LCX_0.3_ant_train_S69_000001"]
        assert lcx["mi_occlusion_site"] == "LCX"
        assert lcx["mi_transmurality"] == 0.3
        assert lcx["mi_region"] == "ant"
        # RCA is not split anterior/posterior — only LCX is.
        assert pd.isna(df.loc["mi_RCA_1.0_test_S62_000001", "mi_region"])
        # Non-MI records carry none of the three.
        sinus = df.loc["sinus_train_S65_000001"]
        assert pd.isna(sinus["mi_subclass"])
        assert pd.isna(sinus["mi_occlusion_site"])
        assert pd.isna(sinus["mi_transmurality"])

    def test_pathology_subclass_is_the_fifteen_class_label(self, tmp_path):
        df = self._build(tmp_path)
        assert set(df["pathology_subclass"]) == {
            "sinus", "iab", "mi_LCX_0.3_ant", "mi_RCA_1.0"
        }
        labels = self._splitter().get_stratification_labels(df, None)
        assert labels.tolist() == df["pathology_subclass"].tolist()

    def test_source_split_maps_to_folds_one_two_three(self, tmp_path):
        """The authors' own partition, taken verbatim — `validation` is fold 2."""
        from ecgbench.splitting.strategies.medalcare_xl import SPLIT_TO_FOLD

        assert SPLIT_TO_FOLD == {"train": 1, "validation": 2, "test": 3}
        df = self._build(tmp_path)
        assert dict(zip(df["source_split"], df["fold"], strict=True)) == {
            "train": 1, "validation": 2, "test": 3
        }

    def test_model_id_is_the_ventricular_model_not_a_patient(self, tmp_path):
        """There are no patients; `model_id` groups by simulation model."""
        df = self._build(tmp_path)
        assert set(df["model_id"]) == {"S65", "S64", "S67", "S69", "S62"}
        assert not df["model_id"].str.startswith("run_").any()

    def test_an_unrecognised_mi_subclass_raises(self, tmp_path):
        """A directory that looks like data must parse, not be silently accepted."""
        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.medalcare_xl import SIGNALS_DIR, build_metadata

        root = self._tree(tmp_path)
        bogus = root / SIGNALS_DIR / "mi" / "LAD_0.7_lateral" / "train" / "run_S65"
        bogus.mkdir(parents=True)
        (bogus / "000001_filtered.csv").write_text("0,0\n")
        with pytest.raises(ValueError, match="Unrecognised MI subclass"):
            build_metadata(root, load_config("medalcare_xl"))

    def test_a_missing_signal_tree_names_the_nesting_trap(self, tmp_path):
        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.medalcare_xl import build_metadata

        with pytest.raises(FileNotFoundError, match="MedalCare-XL/MedalCare-XL"):
            build_metadata(tmp_path, load_config("medalcare_xl"))

    def _splitter(self):
        from ecgbench.splitting.strategies.medalcare_xl import MedalCareXLSplitter

        return MedalCareXLSplitter()

    def test_load_metadata_caches_to_disk_for_the_validation_engine(self, tmp_path):
        """validate_dataset re-reads the CSV, so an in-memory frame is not enough."""
        from ecgbench.config import load_config

        config = load_config("medalcare_xl")
        root = self._tree(tmp_path)
        df = self._splitter().load_metadata(root, config)
        cached = root / config.metadata_csv
        assert cached.exists()
        # The second call reads the cache and agrees with the first.
        again = self._splitter().load_metadata(root, config)
        assert again["record_id"].tolist() == df["record_id"].tolist()
        assert again["record_number"].tolist() == df["record_number"].tolist()


class TestMITDBSplitter:
    """48 records, 47 subjects, and one pair that must not be separated."""

    def _header(self, rec, tape, leads=("MLII", "V1"), speed=1):
        return (
            f"{rec} 2 360 650000\n"
            + "".join(
                f"{rec}.dat 212 200 11 1024 1000 0 0 {lead}\n" for lead in leads
            )
            + f"# 68 M {tape} 2851 x{speed}\n"
            "# Digoxin\n"
            "# The PVCs are uniform.\n"
        )

    def _tree(self, tmp_path, records):
        """records: [(rec, tape, leads)] -> flat headers, .atr each, plus RECORDS."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, tape, leads in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(rec, tape, leads), encoding="utf-8"
            )
            wfdb.wrann(
                rec, "atr",
                sample=np.array([100, 200, 300, 400]),
                symbol=["+", "N", "N", "V"],
                aux_note=["(N", "", "", ""],
                fs=360,
                write_dir=str(tmp_path),
            )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, _, _ in records) + "\n", encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="mitdb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="patient_id",
            signal_path_columns={360: "signal_path"}, default_sampling_rate=360,
            label_column="dominant_rhythm", leads=2,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.mitdb import MITDBSplitter

        assert isinstance(get_splitter("mitdb"), MITDBSplitter)

    def test_builds_tape_grouping_signal_paths_and_beat_counts(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        # 201 and 202 came off the same analog tape, which is the whole reason
        # this dataset has 47 subjects and not 48.
        tree = self._tree(
            tmp_path,
            [("201", 1960, ("MLII", "V1")), ("202", 1960, ("MLII", "V1")),
             ("100", 1085, ("MLII", "V1"))],
        )
        df = get_splitter("mitdb").load_metadata(tree, config)

        assert df.loc[df.record_name == "201", "patient_id"].item() == "tape1960"
        assert df.loc[df.record_name == "202", "patient_id"].item() == "tape1960"
        assert df["patient_id"].nunique() == 2  # 3 records, 2 tapes
        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["100", "201", "202"]
        # '+' is a rhythm change, not a beat.
        assert df["n_beats"].tolist() == [3, 3, 3]
        assert df["beat_V"].tolist() == [1, 1, 1]
        assert df["n_rhythm_changes"].tolist() == [1, 1, 1]
        # Written to disk because validate_dataset re-reads it.
        assert (tree / config.metadata_csv).exists()

    def test_the_per_record_lead_layout_reaches_the_metadata(
        self, sample_config, tmp_path
    ):
        """Record 114 stores V5 then MLII, and a user must be able to see that."""
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [("100", 1085, ("MLII", "V1")), ("114", 750, ("V5", "MLII")),
             ("102", 1525, ("V5", "V2"))],
        )
        df = get_splitter("mitdb").load_metadata(tree, config).set_index("record_name")

        assert df.loc["100", "lead_names"] == "MLII|V1"
        assert df.loc["114", "lead_names"] == "V5|MLII"
        assert df.loc["102", "lead_names"] == "V5|V2"

    def test_stratification_is_the_two_documented_halves(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        from ecgbench.labels.mitdb import RANDOM_SAMPLE, SELECTED

        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [("100", 1085, ("MLII", "V1")), ("124", 1199, ("MLII", "V4")),
             ("200", 1953, ("MLII", "V1")), ("234", 1971, ("MLII", "V1"))],
        )
        splitter = get_splitter("mitdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "record_group"
        assert labels.tolist() == [RANDOM_SAMPLE, RANDOM_SAMPLE, SELECTED, SELECTED]

    def test_missing_stratify_column_is_an_error_not_a_silent_fallback(
        self, sample_config
    ):
        import pandas as pd

        config = self._config(sample_config)
        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("mitdb").get_stratification_labels(
                pd.DataFrame({"record_name": ["100"]}), config
            )

    def test_the_two_records_from_one_tape_never_span_a_fold(
        self, sample_config, tmp_path
    ):
        """The concrete leakage this dataset can produce, and the guard against it.

        Ungrouped, 201 and 202 land in different folds most of the time; grouped,
        never. It is one subject out of 47, but it is the only one there is.
        """
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        records = [("201", 1960, ("MLII", "V1")), ("202", 1960, ("MLII", "V1"))]
        records += [(str(100 + i), 1000 + i, ("MLII", "V1")) for i in range(1, 9)]
        records += [(str(200 + i), 2000 + i, ("MLII", "V1")) for i in range(3, 13)]
        tree = self._tree(tmp_path, records)

        splitter = get_splitter("mitdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=5)

        assigned = pd.concat(
            [frame.assign(fold=number) for number, frame in result.folds.items()],
            ignore_index=True,
        )
        folds = assigned.set_index("record_name")["fold"]
        assert folds["201"] == folds["202"]
        spanning = assigned.groupby("patient_id")["fold"].nunique()
        assert (spanning == 1).all()
        assert result.group_column == "patient_id"


class TestLTAFDBSplitter:
    """84 records, three AF classes, and record ids that must stay strings."""

    def _header(self, rec, n_samples=11059200, gains=("202.429", "202.429")):
        return (
            f"{rec} 2 128 {n_samples} 9:30:00 31/01/2003\n"
            + "".join(
                f"{rec}.dat 16 {gain}/mV 0 0 -1 -8202 0 ECG\n" for gain in gains
            )
        )

    def _tree(self, tmp_path, records):
        """records: [(rec, af_fraction)] -> headers, .atr + .qrs each, plus RECORDS.

        The .atr opens an AFIB episode covering ``af_fraction`` of the record and
        an N episode for the rest, which is what af_burden is computed from.
        """
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        n_samples = 11059200
        for rec, af in records:
            (tmp_path / f"{rec}.hea").write_text(self._header(rec), encoding="utf-8")
            # Clamped so the two episodes stay ordered and non-empty at af=0 and
            # af=1; the resulting burden is then within 1e-5 of the requested one.
            boundary = min(max(int(n_samples * (1.0 - af)), 30), n_samples - 100)
            wfdb.wrann(
                rec, "atr",
                sample=np.array([10, 20, boundary, boundary + 10]),
                symbol=["+", "N", "+", "N"],
                aux_note=["(N", "", "(AFIB", ""],
                fs=128,
                write_dir=str(tmp_path),
            )
            wfdb.wrann(
                rec, "qrs",
                sample=np.array([100, 200, 300]),
                symbol=["N", "T", "N"],
                fs=128,
                write_dir=str(tmp_path),
            )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, _ in records) + "\n", encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ltafdb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={128: "signal_path"}, default_sampling_rate=128,
            label_column="dominant_rhythm", leads=2, zero_padded_identifiers=True,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.ltafdb import LTAFDBSplitter

        assert isinstance(get_splitter("ltafdb"), LTAFDBSplitter)

    def test_builds_burden_signal_paths_and_beat_counts(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("00", 0.0), ("01", 0.5), ("100", 1.0)])

        df = get_splitter("ltafdb").load_metadata(tree, config).set_index("record_name")

        # Leading zeros survive the frame the splitter hands back.
        assert set(df.index) == {"00", "01", "100"}
        assert df.loc["00", "af_burden"] == pytest.approx(0.0, abs=0.001)
        assert df.loc["01", "af_burden"] == pytest.approx(0.5, abs=0.001)
        assert df.loc["100", "af_burden"] == pytest.approx(1.0, abs=0.001)
        assert df.loc["00", "af_class"] == "minimal"
        assert df.loc["01", "af_class"] == "paroxysmal"
        assert df.loc["100", "af_class"] == "sustained"
        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["00", "01", "100"]
        # '+' is a rhythm change, not a beat; the .atr N annotations are the beats.
        assert df["n_beats"].tolist() == [2, 2, 2]
        assert df["n_rhythm_changes"].tolist() == [2, 2, 2]
        # .qrs is summarised separately and never folded into n_beats.
        assert df["n_detections"].tolist() == [2, 2, 2]
        assert df["n_af_terminations"].tolist() == [1, 1, 1]
        # Written to disk because validate_dataset re-reads it.
        assert (tree / config.metadata_csv).exists()

    def test_the_cached_csv_round_trip_keeps_leading_zeros(
        self, sample_config, tmp_path
    ):
        """The second call reads the CSV back, which is where the zeros die.

        Without ``dtype=config.identifier_dtypes()`` record "00" returns as 0 and
        ``data_path / "0"`` is not a file — every record then fails corrupt_header
        for a reason nothing in the traceback mentions.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("00", 0.1), ("08", 0.9), ("122", 0.5)])

        splitter = get_splitter("ltafdb")
        splitter.load_metadata(tree, config)  # writes the cache
        cached = splitter.load_metadata(tree, config)  # reads it back

        assert sorted(cached["record_name"]) == ["00", "08", "122"]
        assert sorted(cached["signal_path"]) == ["00", "08", "122"]

    def test_stratification_is_af_class_itself_not_a_coarsened_copy(
        self, sample_config, tmp_path
    ):
        """84 records can fill three classes over ten folds where afdb's 25 cannot.

        afdb has to stratify on a binary 20% cut because ``sustained`` holds 3 of
        its 25 records. Here the label a reader wants and the label the folds use
        are the same column, which is the arrangement to prefer when counts allow.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("00", 0.0), ("01", 0.5), ("100", 1.0)])

        splitter = get_splitter("ltafdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "af_class"
        assert labels.tolist() == ["minimal", "paroxysmal", "sustained"]
        assert labels.tolist() == df["af_class"].tolist()

    def test_missing_stratify_column_is_an_error_not_a_silent_fallback(
        self, sample_config
    ):
        import pandas as pd

        config = self._config(sample_config)
        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("ltafdb").get_stratification_labels(
                pd.DataFrame({"record_name": ["00"]}), config
            )

    def test_folds_are_ungrouped_because_no_subject_id_ships(
        self, sample_config, tmp_path
    ):
        """No header here carries a comment line, so there is nothing to group on.

        mitdb can group on the analog tape number; this release states nothing at
        all about its subjects, so ``patient_id_column`` is null and the engine
        uses plain StratifiedKFold. That is a fact about the data, not an omission.
        """
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        records = [(f"{i:02d}", 0.0 if i % 3 == 0 else 0.5 if i % 3 == 1 else 1.0)
                   for i in range(15)]
        tree = self._tree(tmp_path, records)

        splitter = get_splitter("ltafdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=5)

        assert result.group_column is None
        assigned = pd.concat(
            [frame.assign(fold=number) for number, frame in result.folds.items()],
            ignore_index=True,
        )
        assert len(assigned) == 15
        # Every class reaches every fold, which is what stratification buys here.
        assert assigned.groupby("fold")["af_class"].nunique().tolist() == [3] * 5


class TestChallenge2017Splitter:
    """Single-lead AF challenge: no metadata file, no identifiers, four classes."""

    def _write(self, root, records):
        """Minimal release tree: RECORDS, headers, and the four REFERENCE files."""
        training = root / "training"
        training.mkdir(parents=True, exist_ok=True)
        for relative, code in records:
            (training / relative).parent.mkdir(parents=True, exist_ok=True)
            name = relative.rsplit("/", 1)[-1]
            (training / f"{relative}.hea").write_text(
                f"{name} 1 300 9000 05:05:15 1/05/2000 \n"
                f"{name}.mat 16+24 1000/mV 16 0 -127 0 0 ECG \n",
                encoding="utf-8",
            )
        (training / "RECORDS").write_text(
            "".join(f"{r}\n" for r, _ in records), encoding="utf-8"
        )
        body = "".join(f"{r},{c}\n" for r, c in records)
        for name in ("REFERENCE.csv", "REFERENCE-v0.csv", "REFERENCE-v1.csv",
                     "REFERENCE-v2.csv", "REFERENCE-v3.csv"):
            (training / name).write_text(body, encoding="utf-8")
        return root

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="challenge2017", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={300: "signal_path"}, default_sampling_rate=300,
            label_column="class_name", leads=1,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.challenge2017 import Challenge2017Splitter

        assert isinstance(get_splitter("challenge2017"), Challenge2017Splitter)

    def test_builds_signal_paths_from_the_subdirectory_tree_and_caches_csv(
        self, sample_config, tmp_path
    ):
        """Records live in A00/…/A08 subdirectories, unlike the flat CPSC mirror."""
        config = self._config(sample_config)
        tree = self._write(tmp_path, [("A00/A00001", "N"), ("A08/A08528", "~")])
        df = get_splitter("challenge2017").load_metadata(tree, config)

        paths = dict(zip(df["record_name"], df["signal_path"]))
        assert paths["A00001"] == "training/A00/A00001.mat"
        assert paths["A08528"] == "training/A08/A08528.mat"
        # Written to disk because validate_dataset re-reads it rather than
        # reusing this frame.
        assert (tree / config.metadata_csv).exists()

    def test_cached_csv_keeps_the_tilde_class_code_as_a_string(
        self, sample_config, tmp_path
    ):
        """"~" survives the CSV round trip — it is a label, not a missing value."""
        config = self._config(sample_config)
        tree = self._write(tmp_path, [("A00/A00001", "~")])
        splitter = get_splitter("challenge2017")

        first = splitter.load_metadata(tree, config)
        cached = splitter.load_metadata(tree, config)  # now reads the cache

        assert first.loc[0, "class_code"] == "~"
        assert cached.loc[0, "class_code"] == "~"
        assert cached.loc[0, "class_name"] == "noisy"
        assert cached.loc[0, "record_name"] == "A00001"

    def test_stratification_uses_the_four_classes_unreduced(
        self, sample_config, tmp_path
    ):
        """Records are single-label and the rarest class has 279 records upstream,
        so nothing is pooled and nothing is reduced."""
        config = self._config(sample_config)
        tree = self._write(tmp_path, [
            ("A00/A00001", "N"), ("A00/A00002", "A"),
            ("A00/A00003", "O"), ("A00/A00004", "~"),
        ])
        splitter = get_splitter("challenge2017")
        labels = splitter.get_stratification_labels(
            splitter.load_metadata(tree, config), config
        )

        assert labels.name == "rhythm_class"
        assert set(labels) == {"normal", "atrial_fibrillation", "other_rhythm", "noisy"}
        assert "OTHER" not in set(labels)

    def test_stratification_needs_load_metadata_first(self, sample_config):
        import pandas as pd

        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("challenge2017").get_stratification_labels(
                pd.DataFrame({"record_name": ["A00001"]}), self._config(sample_config)
            )

    def test_folds_are_ungrouped_because_nothing_identifies_a_contributor(
        self, sample_config, tmp_path
    ):
        """No identifiers ship, and this release does not even claim one record
        per person — the recordings came from members of the public who had bought
        a handheld device. So the engine uses plain StratifiedKFold, and a repeat
        contributor would straddle folds undetectably. That is a fact about the
        data, and the dataset page says so.
        """
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        codes = ["N", "A", "O", "~"]
        records = [(f"A00/A{i:05d}", codes[i % 4]) for i in range(20)]
        tree = self._write(tmp_path, records)

        splitter = get_splitter("challenge2017")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=5)

        assert result.group_column is None
        assigned = pd.concat(
            [frame.assign(fold=number) for number, frame in result.folds.items()],
            ignore_index=True,
        )
        assert len(assigned) == 20
        # Every class reaches every fold, which is what stratification buys here.
        assert assigned.groupby("fold")["class_name"].nunique().tolist() == [4] * 5

    def test_missing_release_raises_rather_than_splitting_nothing(
        self, sample_config, tmp_path
    ):
        from ecgbench.labels import LabelSourceMissingError

        with pytest.raises(LabelSourceMissingError, match="RECORDS"):
            get_splitter("challenge2017").load_metadata(
                tmp_path, self._config(sample_config)
            )


class TestNSRDBSplitter:
    """18 healthy controls, one cohort class, and folds balanced on sex instead."""

    def _header(self, rec, n_samples, age, sex):
        """A real NSRDB header: gain 0 (uncalibrated) and one `# age sex` comment."""
        return (
            f"{rec} 2 128 {n_samples}  8:04:00\n"
            f"{rec}.dat 212 0 12 0 -33 15756 0 ECG1\n"
            f"{rec}.dat 212 0 12 0 -65 -21174 0 ECG2\n"
            f"# {age} {sex}\n"
        )

    def _tree(self, tmp_path, records, n_samples=1_280_000):
        """records: [(rec, age, sex, n_ectopic)] -> headers, .atr each, RECORDS.

        The .atr carries beats stopping well before the end of the record — the
        unannotated tail every real record has — plus one `~` transition into
        noise and back, and one `|` isolated-artifact marker.
        """
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, age, sex, n_ectopic in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(rec, n_samples, age, sex), encoding="utf-8"
            )
            # 100 beats exactly 1 s apart (128 samples at 128 Hz), the first at
            # 1280 and the last at 13952 of 1280000 — so almost all of this
            # record has no beat annotation behind it, as in the real release.
            beats = np.arange(100) * 128 + 1280
            symbols = ["N"] * 100
            for i in range(n_ectopic):
                symbols[i + 1] = "V"
            # A noisy interval on ECG2 (subtype 2) covering 1280 samples = 10 s,
            # then a return to clean.
            sample = np.concatenate([[1], beats, [200_000, 201_280]])
            symbol = ["|"] + symbols + ["~", "~"]
            wfdb.wrann(
                rec, "atr",
                sample=sample,
                symbol=symbol,
                subtype=np.array([0] + [0] * 100 + [2, 0]),
                fs=128,
                write_dir=str(tmp_path),
            )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, _, _, _ in records) + "\n", encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="nsrdb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={128: "signal_path"}, default_sampling_rate=128,
            label_column="cohort_label", leads=2, zero_padded_identifiers=False,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.nsrdb import NSRDBSplitter

        assert isinstance(get_splitter("nsrdb"), NSRDBSplitter)

    def test_builds_demographics_beats_quality_and_signal_paths(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path, [("16265", 32, "M", 3), ("16272", 20, "F", 0)]
        )

        df = get_splitter("nsrdb").load_metadata(tree, config).set_index("record_name")

        # The one header comment line is the entire shipped metadata.
        assert df.loc["16265", "age"] == 32
        assert df.loc["16265", "sex"] == "M"
        assert df.loc["16272", "sex"] == "F"
        # Beats, and ectopy counted apart from them.
        assert df.loc["16265", "n_beats"] == 100
        assert df.loc["16265", "n_ectopic_beats"] == 3
        assert df.loc["16272", "n_ectopic_beats"] == 0
        # '|' and '~' are markers, not beats, and must never reach n_beats.
        assert df.loc["16265", "n_isolated_artifacts"] == 1
        assert df.loc["16265", "n_quality_changes"] == 2
        # Signal quality as seconds: 10 s of ECG2-noisy, the rest clean.
        assert df.loc["16265", "noisy_ECG2_secs"] == pytest.approx(10.0)
        assert df.loc["16265", "noisy_ECG1_secs"] == 0.0
        assert df.loc["16265", "clean_secs"] == pytest.approx(1_280_000 / 128 - 10.0)
        # One class for every record — asserted by the release, not derived.
        assert set(df["cohort_label"]) == {"normal_sinus_rhythm"}
        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["16265", "16272"]
        # Written to disk because validate_dataset re-reads it.
        assert (tree / config.metadata_csv).exists()

    def test_the_unannotated_tail_is_measured_rather_than_assumed(
        self, sample_config, tmp_path
    ):
        """Beat annotation covers 79.5%-95.7% of a real record, silently.

        Nothing in the header or the annotation file says the recording continues
        past the last beat, so a window placed by record length alone can land in
        a span with no reference behind it. These columns are how a user avoids it.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("16265", 32, "M", 0)])

        df = get_splitter("nsrdb").load_metadata(tree, config).set_index("record_name")

        # Beats run 1280 -> 13952 of 1280000 samples at 128 Hz.
        assert df.loc["16265", "annotated_secs"] == pytest.approx(99.0)
        assert df.loc["16265", "unannotated_head_secs"] == pytest.approx(10.0)
        assert df.loc["16265", "unannotated_tail_secs"] == pytest.approx(
            (1_280_000 - 13_952) / 128
        )
        assert df.loc["16265", "annotated_fraction"] < 0.01

    def test_hrv_summaries_reject_the_unannotated_gap(self, sample_config, tmp_path):
        """Without the RR filter, the multi-hour tail enters as one RR interval.

        The beats here are exactly 1 s apart, so mean_hr_bpm must be 60 and the
        SD zero. Both only hold if the gap from the last beat to the end of the
        record is excluded — it is not an RR interval at all.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("16265", 32, "M", 0)])

        df = get_splitter("nsrdb").load_metadata(tree, config).set_index("record_name")

        assert df.loc["16265", "mean_hr_bpm"] == pytest.approx(60.0)
        assert df.loc["16265", "sdnn_ms"] == pytest.approx(0.0, abs=1e-6)
        assert df.loc["16265", "rmssd_ms"] == pytest.approx(0.0, abs=1e-6)

    def test_stratification_is_sex_because_there_is_no_clinical_label(
        self, sample_config, tmp_path
    ):
        """cohort_label is one value, so it cannot balance a fold; sex can.

        This is the opposite arrangement to ltafdb, where the label a reader wants
        and the label the folds use are the same column. Here the label a reader
        wants is a constant, so the two must differ — and the fold label is
        demographic, not clinical.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [("16265", 32, "M", 0), ("16272", 20, "F", 0), ("16273", 28, "F", 0)],
        )

        splitter = get_splitter("nsrdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "sex"
        assert labels.tolist() == ["M", "F", "F"]
        assert df["cohort_label"].nunique() == 1

    def test_missing_stratify_column_is_an_error_not_a_silent_fallback(
        self, sample_config
    ):
        import pandas as pd

        config = self._config(sample_config)
        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("nsrdb").get_stratification_labels(
                pd.DataFrame({"record_name": ["16265"]}), config
            )

    def test_folds_are_ungrouped_because_no_subject_id_ships(
        self, sample_config, tmp_path
    ):
        """The header comment holds age and sex — no tape, recorder or subject id.

        mitdb can group on the analog tape number; this release states nothing
        that would tie two records to one person, and PhysioNet describes 18
        recordings from 18 subjects, so ``patient_id_column`` is null and the
        engine uses plain StratifiedKFold.
        """
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        records = [
            (f"1{i:04d}", 20 + i, "M" if i % 3 == 0 else "F", 0) for i in range(15)
        ]
        tree = self._tree(tmp_path, records)

        splitter = get_splitter("nsrdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=5)

        assert result.group_column is None
        assigned = pd.concat(
            [frame.assign(fold=number) for number, frame in result.folds.items()],
            ignore_index=True,
        )
        assert len(assigned) == 15
        # Both sexes reach every fold, which is what stratifying on it buys.
        assert assigned.groupby("fold")["sex"].nunique().tolist() == [2] * 5


class TestCHFDBSplitter:
    """15 severe-CHF subjects, one severity class, folds balanced on sex instead."""

    def _header(self, rec, n_samples, age, sex):
        """A real CHFDB header: gain 0 (uncalibrated) and one NYHA comment line."""
        return (
            f"{rec} 2 250 {n_samples} 10:00:00\n"
            f"{rec}.dat 212 0 12 0 127 17579 0 ECG1\n"
            f"{rec}.dat 212 0 12 0 -128 21162 0 ECG2\n"
            f"#Age: {age}  Sex: {sex}  NYHA class: III-IV\n"
        )

    def _tree(self, tmp_path, records, n_samples=250_000):
        """records: [(rec, age, sex, n_ront, af)] -> headers, .ecg each, RECORDS.

        Unlike the nsrdb fixture the beats span the whole record, because chfdb's
        annotations do. ``n_ront`` writes `r` beats — R-on-T PVCs, which are AAMI
        ventricular and are the symbol that outnumbers plain `V` in most real
        records. ``af`` adds a `(AF`/`(N` rhythm pair.
        """
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, age, sex, n_ront, af in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(rec, n_samples, age, sex), encoding="utf-8"
            )
            # 100 beats exactly 1 s apart (250 samples at 250 Hz), the first at 250
            # and the last at 25000 of 250000 — the fixture is short, but the point
            # is that nothing is left unannotated at either end by construction.
            beats = np.arange(100) * 250 + 250
            symbols = ["N"] * 100
            for i in range(n_ront):
                symbols[i + 1] = "r"
            sample = beats
            symbol = list(symbols)
            subtype = [0] * 100
            aux = [""] * 100
            if af:
                # AF from 100 s to 110 s, then back to normal.
                sample = np.concatenate([beats, [25_000, 27_500]])
                symbol = symbols + ["+", "+"]
                subtype = [0] * 102
                aux = [""] * 100 + ["(AF", "(N"]
            wfdb.wrann(
                rec, "ecg",
                sample=np.asarray(sample),
                symbol=symbol,
                subtype=np.array(subtype),
                aux_note=aux,
                fs=250,
                write_dir=str(tmp_path),
            )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, *_ in records) + "\n", encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="chfdb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={250: "signal_path"}, default_sampling_rate=250,
            label_column="cohort_label", leads=2, zero_padded_identifiers=False,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.chfdb import CHFDBSplitter

        assert isinstance(get_splitter("chfdb"), CHFDBSplitter)

    def test_builds_demographics_nyha_beats_and_signal_paths(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path, [("chf01", 71, "M", 3, False), ("chf02", 61, "F", 0, False)]
        )

        df = get_splitter("chfdb").load_metadata(tree, config).set_index("record_name")

        # The one header comment line is the entire shipped metadata — and it
        # carries NYHA class, which nsrdb's does not.
        assert df.loc["chf01", "age"] == 71
        assert df.loc["chf01", "sex"] == "M"
        assert df.loc["chf02", "sex"] == "F"
        assert set(df["nyha_class"]) == {"III-IV"}
        # Beats, with `r` reduced into the AAMI ventricular class.
        assert df.loc["chf01", "n_beats"] == 100
        assert df.loc["chf01", "beat_r"] == 3
        assert df.loc["chf01", "aami_V"] == 3
        assert df.loc["chf01", "n_veb"] == 3
        assert df.loc["chf02", "n_veb"] == 0
        # One class for every record — asserted by the release, not derived.
        assert set(df["cohort_label"]) == {"severe_chf"}
        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["chf01", "chf02"]
        # Written to disk because validate_dataset re-reads it.
        assert (tree / config.metadata_csv).exists()

    def test_an_unknown_age_survives_the_csv_round_trip_as_nan(
        self, sample_config, tmp_path
    ):
        """chf06's header records `Age: ?`, and the cache must not turn that into 0."""
        pytest.importorskip("wfdb")
        import numpy as np

        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("chf06", "?", "M", 0, False)])

        splitter = get_splitter("chfdb")
        first = splitter.load_metadata(tree, config).set_index("record_name")
        # Second call reads the CSV it just wrote, which is the path that matters.
        cached = splitter.load_metadata(tree, config).set_index("record_name")

        assert np.isnan(first.loc["chf06", "age"])
        assert np.isnan(cached.loc["chf06", "age"])
        assert cached.loc["chf06", "sex"] == "M"

    def test_beat_annotation_covers_the_record_unlike_nsrdb(
        self, sample_config, tmp_path
    ):
        """Real chfdb heads are 0.05-0.60 s and tails under 0.65 s, in all 15.

        This is the opposite of nsrdb, where one to five hours of every record has
        no beat annotation behind it. The columns are reported anyway so that a
        re-release cannot change it quietly.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("chf01", 71, "M", 0, False)])

        df = get_splitter("chfdb").load_metadata(tree, config).set_index("record_name")

        # Beats run 250 -> 25000 of 250000 samples at 250 Hz.
        assert df.loc["chf01", "annotated_secs"] == pytest.approx(99.0)
        assert df.loc["chf01", "unannotated_head_secs"] == pytest.approx(1.0)

    def test_rhythm_annotation_is_present_only_where_markers_are(
        self, sample_config, tmp_path
    ):
        """11 of the 15 real records carry no `+`, and that is not "no AF".

        A user filtering on ``af_secs == 0`` would otherwise pick up 11 records
        whose rhythm was never assessed as though they were confirmed negatives.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path, [("chf06", 61, "M", 0, True), ("chf02", 61, "F", 0, False)]
        )

        df = get_splitter("chfdb").load_metadata(tree, config).set_index("record_name")

        assert bool(df.loc["chf06", "has_rhythm_annotation"]) is True
        assert df.loc["chf06", "af_secs"] == pytest.approx(10.0)
        assert df.loc["chf06", "n_af_episodes"] == 1
        # No markers at all: zero AF seconds, but no rhythm assertion either.
        assert bool(df.loc["chf02", "has_rhythm_annotation"]) is False
        assert df.loc["chf02", "af_secs"] == 0.0

    def test_hrv_summaries_reject_implausible_intervals(self, sample_config, tmp_path):
        """Beats exactly 1 s apart must give 60 bpm and zero variability."""
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [("chf01", 71, "M", 0, False)])

        df = get_splitter("chfdb").load_metadata(tree, config).set_index("record_name")

        assert df.loc["chf01", "mean_hr_bpm"] == pytest.approx(60.0)
        assert df.loc["chf01", "sdnn_ms"] == pytest.approx(0.0, abs=1e-6)
        assert df.loc["chf01", "rmssd_ms"] == pytest.approx(0.0, abs=1e-6)

    def test_stratification_is_sex_because_every_record_is_the_same_severity(
        self, sample_config, tmp_path
    ):
        """cohort_label and nyha_class are both constants, so neither can balance a fold.

        Same arrangement as nsrdb and the opposite of ltafdb, where the label a
        reader wants and the label the folds use are one column. Here the clinical
        label is a constant, so the fold label has to be demographic.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [("chf01", 71, "M", 0, False), ("chf02", 61, "F", 0, False),
             ("chf03", 63, "M", 0, False)],
        )

        splitter = get_splitter("chfdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "sex"
        assert labels.tolist() == ["M", "F", "M"]
        assert df["cohort_label"].nunique() == 1
        assert df["nyha_class"].nunique() == 1

    def test_missing_stratify_column_is_an_error_not_a_silent_fallback(
        self, sample_config
    ):
        import pandas as pd

        config = self._config(sample_config)
        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("chfdb").get_stratification_labels(
                pd.DataFrame({"record_name": ["chf01"]}), config
            )

    def test_folds_are_ungrouped_because_no_subject_id_ships(
        self, sample_config, tmp_path
    ):
        """The header comment holds age, sex and NYHA class — no tape or subject id.

        mitdb can group on the analog tape number; this release states nothing that
        would tie two records to one person — not even the trial arm the cohort is
        defined by — and PhysioNet describes 15 recordings from 15 subjects, so
        ``patient_id_column`` is null and the engine uses plain StratifiedKFold.
        """
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        # 10 men and 5 women: the real release is 11/4, and both clear the
        # StratifiedKFold requirement that one class hold at least n_folds members.
        records = [
            (f"chf{i + 1:02d}", 50 + i, "M" if i % 3 else "F", 0, False)
            for i in range(15)
        ]
        tree = self._tree(tmp_path, records)

        splitter = get_splitter("chfdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=10)

        assert result.group_column is None
        assigned = pd.concat(
            [frame.assign(fold=number) for number, frame in result.folds.items()],
            ignore_index=True,
        )
        assert len(assigned) == 15
        assert sorted(result.folds) == list(range(1, 11))


class TestSDDBSplitter:
    """23 sudden-cardiac-death subjects: two annotators, an off-file clinical table."""

    def _header(self, rec, n_samples, gain=800, vfon="07:54:33"):
        """A real SDDB header: BOTH channels described as bare "ECG", plus #vfon:."""
        lines = [
            f"{rec} 2 250 {n_samples} 12:00:00",
            f"{rec}.dat 212 {gain} 12 0 51 -24065 0 ECG",
            f"{rec}.dat 212 {gain} 12 0 145 21051 0 ECG",
            f"#Produced by xform_new from record {rec}, beginning at 26:35.000",
        ]
        if vfon:
            lines.append(f"#vfon: {vfon}")
        return "\n".join(lines) + "\n"

    def _tree(self, tmp_path, records, n_samples=250_000):
        """records: [(rec, gain, vfon, audited)] -> header, .ari, optional .atr, RECORDS.

        Every record gets a `.ari` (all 23 real ones do) opening with 50 `?` LEARN
        annotations, which is the shape of the real files. Only ``audited`` records
        get a `.atr`, because 11 of the 23 have none — the asymmetry the prefixed
        columns exist for. The `.atr` uses `/` and `f`, which the `.ari` never does.
        """
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, gain, vfon, audited in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(rec, n_samples, gain, vfon), encoding="utf-8"
            )
            # 50 learning annotations, then 100 beats 1 s apart with three R-on-T.
            learn = np.arange(50) * 100 + 100
            beats = np.arange(100) * 250 + 10_000
            symbols = ["?"] * 50 + ["N"] * 100
            for i in range(3):
                symbols[50 + i + 1] = "r"
            wfdb.wrann(
                rec, "ari",
                sample=np.concatenate([learn, beats]),
                symbol=symbols,
                subtype=np.zeros(150, dtype=int),
                aux_note=[""] * 150,
                fs=250, write_dir=str(tmp_path),
            )
            if audited:
                # Paced beats and one paced/normal fusion — symbols the .ari lacks.
                wfdb.wrann(
                    rec, "atr",
                    sample=beats,
                    symbol=["/"] * 99 + ["f"],
                    subtype=np.zeros(100, dtype=int),
                    aux_note=[""] * 100,
                    fs=250, write_dir=str(tmp_path),
                )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, *_ in records) + "\n", encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="sddb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column=None,
            signal_path_columns={250: "signal_path"}, default_sampling_rate=250,
            label_column="rhythm_class", leads=2, zero_padded_identifiers=False,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.sddb import SDDBSplitter

        assert isinstance(get_splitter("sddb"), SDDBSplitter)

    def test_builds_the_clinical_table_both_annotators_and_signal_paths(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        # 30 has an audited .atr; 33 does not. 47 declares the low gain.
        tree = self._tree(
            tmp_path,
            [("30", 800, "07:54:33", True), ("33", 800, "04:46:19", False),
             ("47", 200, "06:13:01", False)],
        )

        df = get_splitter("sddb").load_metadata(tree, config).set_index("record_name")

        # The clinical table is transcribed from the landing page, not the files.
        assert df.loc["30", "sex"] == "M" and df.loc["30", "age"] == 43
        assert df.loc["33", "sex"] == "F" and df.loc["33", "age"] == 30
        assert set(df["rhythm_class"]) == {"sinus"}
        # The per-record gain, which is 200 for 47 and 800 for the rest.
        assert df.loc["47", "adc_gain"] == "200|200"
        assert df.loc["30", "adc_gain"] == "800|800"
        # Both channels are described as bare "ECG" in the files themselves.
        assert set(df["lead_names"]) == {"ECG|ECG"}
        # Two annotators, kept apart. Only 30 has the audited one.
        assert bool(df.loc["30", "has_audited_annotation"]) is True
        assert bool(df.loc["33", "has_audited_annotation"]) is False
        assert df.loc["30", "atr_n_beats"] == 100
        assert df.loc["33", "atr_n_beats"] == 0
        # The unaudited file's 50 LEARN annotations are NOT beats.
        assert df.loc["30", "ari_n_beats"] == 100
        assert df.loc["30", "ari_n_learning"] == 50
        assert df.loc["30", "ari_beat_r"] == 3
        assert df.loc["30", "ari_aami_V"] == 3
        # And the two vocabularies reduce differently: / and f are AAMI Q.
        assert df.loc["30", "atr_aami_Q"] == 100
        assert df.loc["30", "atr_n_paced_beats"] == 100
        # One class for every record, asserted by the release rather than derived.
        assert set(df["cohort_label"]) == {"sudden_cardiac_death"}
        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["30", "33", "47"]
        # Written to disk because validate_dataset re-reads it.
        assert (tree / config.metadata_csv).exists()

    def test_the_vf_onset_survives_the_csv_round_trip_and_may_be_absent(
        self, sample_config, tmp_path
    ):
        """Records 40, 42 and 49 have no `#vfon:`, and that must stay NaN not 0.

        Zero would claim the terminal arrhythmia began at the first sample of the
        recording, which is both wrong and exactly the kind of value a downstream
        mean would swallow.
        """
        pytest.importorskip("wfdb")
        import numpy as np

        config = self._config(sample_config)
        tree = self._tree(
            tmp_path, [("30", 800, "07:54:33", False), ("42", 800, None, False)]
        )

        splitter = get_splitter("sddb")
        first = splitter.load_metadata(tree, config).set_index("record_name")
        # Second call reads the CSV it just wrote, which is the path that matters.
        cached = splitter.load_metadata(tree, config).set_index("record_name")

        for df in (first, cached):
            assert df.loc["30", "vf_onset_secs"] == 28473.0
            assert bool(df.loc["30", "has_vf_onset"]) is True
            assert np.isnan(df.loc["42", "vf_onset_secs"])
            assert bool(df.loc["42", "has_vf_onset"]) is False

    def test_stratification_labels_come_from_the_loader_not_a_second_derivation(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(
            tmp_path,
            [("30", 800, "07:54:33", False),   # sinus
             ("35", 800, "24:34:56", False),   # atrial fibrillation
             ("40", 800, None, False)],        # continuously paced
        )

        splitter = get_splitter("sddb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "rhythm_class"
        assert labels.tolist() == ["sinus", "afib", "paced"]

    def test_missing_stratify_column_names_the_fix(self, sample_config, tmp_path):
        import pandas as pd

        config = self._config(sample_config)
        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("sddb").get_stratification_labels(
                pd.DataFrame({"record_name": ["30"]}), config
            )


class TestQTDBSplitter:
    """105 delineation excerpts: provenance-stratified folds, two shared subjects."""

    def _header(self, rec, source, offset, n_samples, leads, gain="200",
                counter=None, clinical=(), delay=None):
        rate = f"250/{counter}" if counter else "250"
        # The four records that declare an explicit baseline are also the four with
        # 11-bit resolution and an adc_zero of 1024 — that combination is what makes
        # the baseline meaningful, so the fixture keeps them together.
        res, zero = ("11", "1024") if "(" in gain else ("12", "0")
        lines = [
            f"{rec} 2 {rate} {n_samples}",
            f"{rec}.dat 212 {gain} {res} {zero} 13 -10702 0 {leads[0]}",
            f"{rec}.dat 212 {gain} {res} {zero} 20 -30717 0 {leads[1]}",
        ]
        lines += [f"#{line}" for line in clinical]
        lines.append(f"#Produced by xform from record {source}, beginning at {offset}")
        if delay:
            lines.append(f"#The signal 0 was delayed with a delay={delay} samples")
        return "\n".join(lines) + "\n"

    def _tree(self, tmp_path, records):
        """records: [(rec, source, offset, n_samples, leads, gain, counter, clinical)].

        Every record gets a `.q1c` with two annotated beats and a `.man`; only some
        get a `.atr`, because the 23 sudden-death excerpts have none. Written under a
        letter-only extension and renamed, since ``wfdb.wrann`` refuses ``q1c``.
        """
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, source, offset, n_samples, leads, gain, counter, clinical, atr in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(
                    rec, source, offset, n_samples, leads, gain, counter, clinical,
                    delay=7 if n_samples == 224993 else None,
                ),
                encoding="utf-8",
            )
            samples, symbols, nums = [], [], []
            for start in (150000, 150250):
                for off, sym, num in (
                    (0, "(", 0), (10, "p", 0), (20, ")", 0),
                    (40, "(", 1), (50, "N", 1), (62, ")", 1),
                    (100, "t", 2), (144, ")", 2),
                ):
                    samples.append(start + off)
                    symbols.append(sym)
                    nums.append(num)
            wfdb.wrann(
                rec, "qxc", sample=np.array(samples), symbol=symbols,
                num=np.array(nums), fs=250, write_dir=str(tmp_path),
            )
            (tmp_path / f"{rec}.qxc").rename(tmp_path / f"{rec}.q1c")
            wfdb.wrann(
                rec, "man", sample=np.array([150050, 150300]), symbol=["N", "N"],
                fs=250, write_dir=str(tmp_path),
            )
            if atr:
                wfdb.wrann(
                    rec, "atr", sample=np.arange(100) * 250 + 1000,
                    symbol=["N"] * 99 + ["V"], fs=250, write_dir=str(tmp_path),
                )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, *_ in records) + "\n", encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="qtdb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="patient_id",
            signal_path_columns={250: "signal_path"}, default_sampling_rate=250,
            label_column="source_database", leads=2, zero_padded_identifiers=False,
        )

    #: (rec, source, offset, n_samples, leads, gain, counter, clinical, has_atr)
    MITDB = ("sel100", "100", "7:00.000", 225000, ("MLII", "V5"), "200(0)", "360",
             ("69 M 1085 1629 x1", "Aldomet, Inderal"), True)
    SDDB = ("sel30", "30", "7:39:30.000", 224993, ("ECG1", "ECG2"), "200", None,
            (), False)
    EDB_A = ("sele0121", "e0121", "1:07:30.000", 225000, ("V4", "D3"), "200", None,
             ("Age: 51  Sex: M", "Coronary artery disease"), True)
    EDB_B = ("sele0122", "e0122", "40:00.000", 225000, ("V4", "D3"), "200", None,
             ("Age: 51  Sex: M", "Coronary artery disease"), True)
    LTDB = ("sel14046", "14046", "9:14:13.000", 224999, ("ECG1", "ECG2"), "0", "128",
            (), True)

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.qtdb import QTDBSplitter

        assert isinstance(get_splitter("qtdb"), QTDBSplitter)

    def test_builds_provenance_annotation_summaries_and_signal_paths(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [self.MITDB, self.SDDB, self.LTDB])

        df = get_splitter("qtdb").load_metadata(tree, config).set_index("record_name")

        # Provenance: the source database is the release's own Table 1 stratum, and
        # the source record and offset come from the header's xform line.
        assert df.loc["sel100", "source_database"] == "mitdb"
        assert df.loc["sel100", "source_record"] == "100"
        assert df.loc["sel100", "source_offset_secs"] == 420.0
        assert df.loc["sel100", "source_sampling_rate"] == 360
        assert bool(df.loc["sel100", "resampled_from_source"]) is True
        # sddb is already 250 Hz, so no counter frequency and no resampling.
        assert df.loc["sel30", "source_database"] == "sddb"
        assert bool(df.loc["sel30", "resampled_from_source"]) is False
        assert df.loc["sel30", "signal_0_delay_samples"] == 7
        assert df.loc["sel30", "n_samples"] == 224993
        # The leakage partner, which is what makes the overlap actionable.
        assert df.loc["sel100", "source_catalogue_slug"] == "mit-bih-arrhythmia-database"
        assert df.loc["sel14046", "source_catalogue_slug"] == ""   # ltdb is not in it

        # Manual boundaries: two beats each, both with a QT and a P wave, no U wave.
        assert df.loc["sel100", "n_annotated_beats"] == 2
        assert df.loc["sel100", "n_p_waves"] == 2
        assert df.loc["sel100", "n_t_ends"] == 2
        assert df.loc["sel100", "n_u_waves"] == 0
        assert df.loc["sel100", "median_qt_ms"] == pytest.approx(416.0)
        assert df.loc["sel100", "waveform_pattern"] == "(p)(N)t)"
        # Which is the paper's own pattern for this record, so the check passes.
        assert bool(df.loc["sel100", "waveform_pattern_matches_published"]) is True

        # Inherited .atr, absent for the sudden-death excerpt.
        assert bool(df.loc["sel100", "has_source_annotations"]) is True
        assert bool(df.loc["sel30", "has_source_annotations"]) is False
        assert df.loc["sel100", "n_source_beats"] == 100
        assert df.loc["sel100", "n_source_aami_V"] == 1

        # Calibration: the +5.12 mV pedestal on sel100, and ltdb's declared gain of 0.
        assert df.loc["sel100", "dc_pedestal_mv"] == pytest.approx(5.12)
        assert df.loc["sel30", "dc_pedestal_mv"] == 0.0
        assert df.loc["sel14046", "declared_gain_0"] == 0.0
        assert bool(df.loc["sel14046", "amplitude_calibrated"]) is False
        assert bool(df.loc["sel30", "amplitude_calibrated"]) is False   # an estimate
        assert bool(df.loc["sel100", "amplitude_calibrated"]) is True

        # Per-record lead names, and the flag for the 57 that name no anatomy.
        assert df.loc["sel100", "lead_names"] == "MLII;V5"
        assert bool(df.loc["sel100", "positional_lead_names"]) is False
        assert bool(df.loc["sel30", "positional_lead_names"]) is True

        # Flat tree: the signal path is the bare stem, no extension, no directory.
        assert sorted(df["signal_path"]) == ["sel100", "sel14046", "sel30"]
        # Written to disk because validate_dataset re-reads it.
        assert (tree / config.metadata_csv).exists()

    def test_the_two_shared_edb_subjects_survive_the_csv_round_trip(
        self, sample_config, tmp_path
    ):
        """`sele0121` and `sele0122` are one man, and the group id must not be lost.

        The second call reads the CSV the first one wrote, which is the path
        ``validate_dataset`` and every later run take — a patient id that only exists
        in memory would put the same person in train and test.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [self.MITDB, self.EDB_A, self.EDB_B])

        splitter = get_splitter("qtdb")
        first = splitter.load_metadata(tree, config).set_index("record_name")
        cached = splitter.load_metadata(tree, config).set_index("record_name")

        for df in (first, cached):
            assert df.loc["sele0121", "patient_id"] == "sele0121"
            assert df.loc["sele0122", "patient_id"] == "sele0121"
            assert df.loc["sel100", "patient_id"] == "sel100"
            # Three records, two subjects.
            assert df["patient_id"].nunique() == 2
            # The ESC clinical vintage, which is coarser than edb's own.
            assert df.loc["sele0121", "clinical_source"] == "esc_header"
            assert df.loc["sele0121", "age"] == 51.0
            assert df.loc["sele0121", "clinical_findings"] == "Coronary artery disease"

    def test_stratification_labels_come_from_the_loader_not_a_second_derivation(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, [self.MITDB, self.SDDB, self.EDB_A, self.LTDB])

        splitter = get_splitter("qtdb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "source_database"
        assert labels.tolist() == ["mitdb", "sddb", "edb", "ltdb"]

    def test_missing_stratify_column_names_the_fix(self, sample_config, tmp_path):
        import pandas as pd

        config = self._config(sample_config)
        with pytest.raises(ValueError, match="load_metadata"):
            get_splitter("qtdb").get_stratification_labels(
                pd.DataFrame({"record_name": ["sel100"]}), config
            )

    def test_a_record_outside_the_papers_table_is_refused_rather_than_guessed(
        self, sample_config, tmp_path
    ):
        """The source database cannot be derived from the record name, so it must fail.

        ``sel17152`` and ``sel17453`` differ only in their last three digits and come
        from different sources, so any rule that infers the source from the name is
        wrong for at least one record. A release with a record the transcribed table
        does not cover has to raise rather than fall back.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        unknown = ("sel999", "999", "0", 225000, ("ECG1", "ECG2"), "200", None,
                   (), False)
        tree = self._tree(tmp_path, [self.MITDB, unknown])

        with pytest.raises(ValueError, match="SOURCE_DATABASE_RECORDS"):
            get_splitter("qtdb").load_metadata(tree, config)


class TestSHDBAFSplitter:
    """128 Japanese Holters: a real clinical CSV that still cannot drive the split."""

    CLINICAL = (
        "Data_ID,Subject_ID,Annotated,Age_at_Holter,Sex,AF_Type,CHF,HTN\n"
        "001,2043771,True,65,M,PAF,0.0,False\n"
        "005,4899921,True,47,M,PAF,0.0,False\n"
        "020,4899921,True,50,M,PAF,0.0,False\n"
        "066,5133906,False,53,M,non-AF,0.0,True\n"
        "118,5133906,True,53,M,non-AF,0.0,True\n"
        "143,9000001,False,70,F,PerAF,1.0,True\n"
    )

    def _tree(self, tmp_path, records):
        """records: [(rec, n_samples, annotated)] — .qrs always, .atr only if annotated."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, n_samples, annotated in records:
            (tmp_path / f"{rec}.hea").write_text(
                f"{rec} 2 200 {n_samples}\n"
                f"{rec}.dat 16 8105.233566939608(-9270)/mV 16 0 -9267 62791 0 ECG1\n"
                f"{rec}.dat 16 11470.645879660451(543)/mV 16 0 408 55919 0 ECG2\n",
                encoding="utf-8",
            )
            # Beats every second, starting at 1 s — never at sample 0, which
            # wfdb.wrann drops for a comment annotation.
            beats = np.arange(1, 101) * 200
            wfdb.wrann(rec, "qrs", sample=beats, symbol=["N"] * len(beats),
                       fs=200, write_dir=str(tmp_path))
            if annotated:
                # Every annotation a '"' comment, the rhythm code only on the first
                # beat of each interval: the layout this dataset actually uses.
                aux = ["(AFIB"] + [""] * 49 + ["(N"] + [""] * 49
                wfdb.wrann(rec, "atr", sample=beats, symbol=['"'] * len(beats),
                           aux_note=aux, fs=200, write_dir=str(tmp_path))
        (tmp_path / "RECORDS.txt").write_text(
            "\n".join(r for r, _, _ in records) + "\n", encoding="utf-8"
        )
        (tmp_path / "AdditionalData.csv").write_text(self.CLINICAL, encoding="utf-8")
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="shdb_af", metadata_csv="ecgbench_metadata.csv",
            record_id_column="Data_ID", patient_id_column="Subject_ID",
            signal_path_columns={200: "signal_path"}, default_sampling_rate=200,
            label_column="AF_Type", leads=2, zero_padded_identifiers=True,
        )

    RECORDS = [("001", 200 * 200, True), ("005", 200 * 200, True),
               ("020", 200 * 200, True), ("066", 200 * 200, False),
               ("118", 200 * 200, True), ("143", 200 * 150, False)]

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.shdb_af import SHDBAFSplitter

        assert isinstance(get_splitter("shdb_af"), SHDBAFSplitter)

    def test_joins_the_clinical_table_to_the_annotations_and_adds_a_signal_path(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, self.RECORDS)

        df = get_splitter("shdb_af").load_metadata(tree, config).set_index("Data_ID")

        assert len(df) == 6
        # The clinical side.
        assert df.loc["001", "AF_Type"] == "PAF"
        assert df.loc["001", "Subject_ID"] == "2043771"
        # The reason a custom splitter is needed at all: no shipped column is this.
        assert df.loc["001", "signal_path"] == "001"
        # The annotation side, forward-filled out of the '"' comments.
        assert df.loc["001", "beats_AFIB"] == 50
        assert df.loc["001", "beats_N"] == 50
        assert df.loc["001", "af_beat_fraction"] == pytest.approx(0.5)
        assert bool(df.loc["001", "has_rhythm_annotation"]) is True
        # And the unannotated records still get a full row.
        assert bool(df.loc["066", "has_rhythm_annotation"]) is False
        assert df.loc["066", "n_detections"] == 100
        assert pd.isna(df.loc["066", "af_burden"])
        # The duplicate pair is flagged in both directions.
        assert df.loc["005", "duplicate_of"] == "020"
        assert df.loc["020", "duplicate_of"] == "005"
        assert df.loc["001", "duplicate_of"] in ("", None) or pd.isna(df.loc["001", "duplicate_of"])

    def test_the_metadata_csv_is_written_to_disk_because_validation_rereads_it(
        self, sample_config, tmp_path
    ):
        """``validate_dataset`` reads ``metadata_csv`` from disk, not this DataFrame.

        An in-memory-only frame leaves validation with no metadata at all — the bug
        Chapman shipped for months, where every record failed ``corrupt_header``.
        """
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, self.RECORDS)

        first = get_splitter("shdb_af").load_metadata(tree, config)
        assert (tree / "ecgbench_metadata.csv").exists()

        # And the second call reads the cache, with the ids still strings.
        second = get_splitter("shdb_af").load_metadata(tree, config)
        assert list(second["Data_ID"]) == list(first["Data_ID"]) == [
            "001", "005", "020", "066", "118", "143"
        ]
        assert list(second["signal_path"]) == ["001", "005", "020", "066", "118", "143"]

    def test_the_fold_label_is_read_not_recomputed(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        config = self._config(sample_config)
        tree = self._tree(tmp_path, self.RECORDS)

        splitter = get_splitter("shdb_af")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "af_type_annotation_class"
        assert list(labels) == [
            "PAF+annotated", "PAF+annotated", "PAF+annotated",
            "non-AF+unannotated", "non-AF+annotated", "PerAF",
        ]

    def test_a_frame_without_the_label_column_is_refused(self, sample_config):
        config = self._config(sample_config)
        df = pd.DataFrame({"Data_ID": ["001"], "AF_Type": ["PAF"]})

        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("shdb_af").get_stratification_labels(df, config)

    def test_folds_keep_the_duplicate_recording_and_its_subject_together(
        self, sample_config, tmp_path
    ):
        """005 and 020 are the same recording; grouping on Subject_ID is what saves it.

        Not a property of the splitter's cleverness — both rows happen to carry
        Subject_ID 4899921. This asserts the mechanism that makes the release's own
        missed duplicate harmless, so a later change to ``patient_id_column`` cannot
        quietly reintroduce the leak.
        """
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        config = self._config(sample_config)
        tree = self._tree(tmp_path, self.RECORDS)
        splitter = get_splitter("shdb_af")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=3, random_state=42)
        assert result.group_column == "Subject_ID"

        fold_of = {
            row["Data_ID"]: fold
            for fold, frame in result.folds.items()
            for _, row in frame.iterrows()
        }
        assert fold_of["005"] == fold_of["020"]
        assert fold_of["066"] == fold_of["118"]


class TestLTSTDBSplitter:
    """86 day-long ST recordings: subject identity in the name, zero-padded.

    The splitting problem here is the opposite of ``stdb``'s. There the release
    identifies its subjects in no way at all; here it identifies them inside the
    record name, so the grouping is published rather than inferred — and the
    identifier it publishes is three zero-padded digits, which a CSV round trip
    turns into an integer unless the config says otherwise.
    """

    #: The real multi-record subjects: 027 has four records, 073/074/075 two each.
    MULTI = {"027": 4, "073": 2, "074": 2, "075": 2}

    def _header(self, rec, n_sig, n_samples, episodes):
        leads = ["ECG", "ECG"] if n_sig == 2 else ["E-S", "A-S", "A-I"]
        lines = [f"{rec} {n_sig} 250 {n_samples} 11:00:00 28/02/1984"]
        for i in range(n_sig):
            lines.append(f"{rec}.dat 212 200/mV 12 0 6 -101 0 {leads[i]}")
        lines += [
            "#Age: 55  Sex: M",
            "#Comments:",
            f"#  {episodes} ischaemic episodes.",
            "#Symptoms during Holter recording: No data",
            "#Diagnoses: ",
            "#  Coronary artery disease",
            "#Treatment:",
            "#  Medications: None",
            "#  Balloon Angioplasty: No",
            "#History: ",
            "#  Hypertension: No",
            "#Holter Recording:",
            "#  Date: 28/02/1984",
            "#  Recorder: Zymed",
        ]
        return "\n".join(lines) + "\n"

    def _tree(self, tmp_path, records):
        """records: [(name, n_sig, n_episodes)] -> headers, .atr, .sta/.stb/.stc, RECORDS."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        n_samples = 5_000_000
        for rec, n_sig, episodes in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(rec, n_sig, n_samples, episodes), encoding="utf-8"
            )
            beats = np.arange(500) * 250 + 1000
            wfdb.wrann(
                rec, "atr",
                sample=beats, symbol=["N"] * 500,
                subtype=np.zeros(500, dtype=int), aux_note=[""] * 500,
                fs=250, write_dir=str(tmp_path),
            )
            sample, aux = [], []
            for i in range(episodes):
                base = 10_000 + i * 3_000
                sample += [base, base + 500, base + 1000]
                aux += [f"(st0-{120 + i}", f"ast0-{200 + i}", f"st0-{60 + i})"]
            if not sample:                       # a record with no ST event at all
                sample, aux = [5_000], ["GRST0"]
            for ext in ("sta", "stb", "stc"):
                wfdb.wrann(
                    rec, ext,
                    sample=np.asarray(sample), symbol=["s"] * len(sample),
                    subtype=np.zeros(len(sample), dtype=int),
                    chan=np.zeros(len(sample), dtype=int),
                    aux_note=aux, fs=250, write_dir=str(tmp_path),
                )
        (tmp_path / "RECORDS").write_text(
            "\n".join(r for r, _, _ in records) + "\n", encoding="utf-8"
        )

    def _records(self):
        """The real 86 names, with a burden spread reproducing the 18/14/25/29 bands."""
        names = [f"s20{n:02d}1" for n in range(1, 27)]                 # s20011..s20261
        names += ["s20271", "s20272", "s20273", "s20274"]              # one subject
        names += [f"s20{n:02d}1" for n in range(28, 66)]               # s20281..s20651
        names += [f"s30{n:02d}1" for n in range(66, 73)]               # s30661..s30721
        names += ["s30731", "s30732", "s30741", "s30742", "s30751", "s30752"]
        names += [f"s30{n:02d}1" for n in range(76, 81)]               # s30761..s30801
        assert len(names) == 86
        # 18 records with none, 14 with 1-5, 25 with 6-20, 29 with 21+.
        burden = [0] * 18 + [3] * 14 + [10] * 25 + [30] * 29
        return [
            (name, 3 if name.startswith("s3") else 2, b)
            for name, b in zip(names, burden)
        ]

    def test_load_metadata_builds_the_csv_the_validator_will_reread(self, tmp_path):
        """Nothing ships to read, so the splitter writes what validate_dataset needs."""
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.ltstdb import LTSTDBSplitter

        self._tree(tmp_path, self._records())
        config = load_config("ltstdb")
        df = LTSTDBSplitter().load_metadata(tmp_path, config)

        assert len(df) == 86
        assert (tmp_path / config.metadata_csv).exists()
        for column in ("record_name", "signal_path", "patient_id", "stratify_class",
                       "n_leads", "n_ischemic_episodes", "n_ischemic_episodes_b"):
            assert column in df.columns
        assert df["record_name"].iloc[0] == "s20011"
        assert df["signal_path"].iloc[0] == "s20011"
        assert df["patient_id"].iloc[0] == "001"

    def test_the_cached_csv_keeps_the_zero_padding_on_patient_id(self, tmp_path):
        """"027" must not come back as 27, or the grouping key stops matching.

        The record name survives a round trip on its own because it starts with a
        letter. The subject number does not, and it is the column folds are grouped
        on — so the reread goes through ``config.identifier_dtypes()``, which is
        non-empty only because the config sets ``zero_padded_identifiers``.
        """
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.ltstdb import LTSTDBSplitter

        self._tree(tmp_path, self._records())
        config = load_config("ltstdb")
        splitter = LTSTDBSplitter()
        splitter.load_metadata(tmp_path, config)
        again = splitter.load_metadata(tmp_path, config)

        assert all(isinstance(v, str) for v in again["patient_id"])
        assert all(isinstance(v, str) for v in again["record_name"])
        assert list(again["patient_id"])[:3] == ["001", "002", "003"]
        assert "027" in set(again["patient_id"])

    def test_the_four_records_of_subject_027_share_a_patient_id(self, tmp_path):
        """s20271-s20274 differ only in the last digit, and the release says so."""
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.ltstdb import LTSTDBSplitter

        self._tree(tmp_path, self._records())
        df = LTSTDBSplitter().load_metadata(tmp_path, load_config("ltstdb"))

        assert df["patient_id"].nunique() == 80
        counts = df["patient_id"].value_counts()
        assert counts[counts > 1].to_dict() == self.MULTI

    def test_stratification_is_the_ischaemic_burden_band(self, tmp_path):
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.ltstdb import LTSTDBSplitter

        self._tree(tmp_path, self._records())
        config = load_config("ltstdb")
        splitter = LTSTDBSplitter()
        df = splitter.load_metadata(tmp_path, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.value_counts().to_dict() == {
            "21+": 29, "6-20": 25, "none": 18, "1-5": 14,
        }
        # Unlike edb, every band clears the 10 folds, so none has to skip any.
        assert labels.value_counts().min() >= 10

    def test_missing_stratify_column_raises_rather_than_silently_restratifying(self):
        import pandas as pd

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.ltstdb import LTSTDBSplitter

        df = pd.DataFrame({"record_name": ["s20011"], "signal_path": ["s20011"]})
        with pytest.raises(ValueError, match="stratify_class"):
            LTSTDBSplitter().get_stratification_labels(df, load_config("ltstdb"))

    def test_the_registry_resolves_ltstdb_to_its_own_splitter(self):
        """A slug mismatch would silently fall back to GenericSplitter."""
        from ecgbench.splitting import get_splitter
        from ecgbench.splitting.strategies.ltstdb import LTSTDBSplitter

        assert isinstance(get_splitter("ltstdb"), LTSTDBSplitter)

    def test_no_subject_spans_two_folds(self, tmp_path):
        """Subject 027's four records hold a quarter of the release's ischaemia.

        Ungrouped they would land in several folds, and any of them appearing in
        both train and test is the same day of the same heart on both sides of the
        split.
        """
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.engine import split_dataset
        from ecgbench.splitting.strategies.ltstdb import LTSTDBSplitter

        config = load_config("ltstdb")
        self._tree(tmp_path, self._records())
        splitter = LTSTDBSplitter()
        df = splitter.load_metadata(tmp_path, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=10)

        assert result.group_column == "patient_id"
        fold_of = {
            record: fold
            for fold, frame in result.folds.items()
            for record in frame["record_name"]
        }
        assert len(fold_of) == 86
        per_subject = {}
        for record, fold in fold_of.items():
            per_subject.setdefault(record[2:5], set()).add(fold)
        spanning = {s: sorted(f) for s, f in per_subject.items() if len(f) > 1}
        assert spanning == {}


class TestSTDBSplitter:
    """28 ST-change recordings: no metadata at all, and two channel layouts."""

    def _header(self, rec, n_samples, n_sig=2, gains=(296, 300)):
        """A real STDB header: no comment line, no start time, bare "ECG" descriptions.

        The absence of a comment line is the point — nsrdb has ``# <age> <sex>``
        and sddb has ``#vfon:``, and this release has nothing to parse at all.
        """
        lines = [f"{rec} {n_sig} 360 {n_samples}"]
        for i in range(n_sig):
            lines.append(f"{rec}.dat 212 {gains[i]} 12 0 40 0 0 ECG")
        return "\n".join(lines) + "\n"

    def _tree(self, tmp_path, records, n_samples=360_000):
        """records: [(rec, n_sig)] -> header, .dat, .atr and a RECORDS file."""
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for rec, n_sig in records:
            (tmp_path / f"{rec}.hea").write_text(
                self._header(rec, n_samples, n_sig), encoding="utf-8"
            )
            beats = np.arange(200) * 360 + 1000
            wfdb.wrann(
                rec, "atr",
                sample=beats,
                symbol=["N"] * 200,
                subtype=np.zeros(200, dtype=int),
                aux_note=[""] * 200,
                fs=360, write_dir=str(tmp_path),
            )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, _ in records) + "\n", encoding="utf-8"
        )

    def _records(self):
        """The real 28: 300-327, with 313-317 and 319-323 holding a single channel."""
        single = {"313", "314", "315", "316", "317", "319", "320", "321", "322", "323"}
        return [(str(n), 1 if str(n) in single else 2) for n in range(300, 328)]

    def test_load_metadata_builds_the_csv_the_validator_will_reread(self, tmp_path):
        """Nothing ships to read, so the splitter writes what validate_dataset needs.

        ``validate_dataset`` re-reads ``metadata_csv`` from disk rather than reusing
        this frame, so an in-memory-only result would leave validation with no
        metadata — the Chapman failure mode.
        """
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.stdb import STDBSplitter

        self._tree(tmp_path, self._records())
        config = load_config("stdb")
        df = STDBSplitter().load_metadata(tmp_path, config)

        assert len(df) == 28
        assert (tmp_path / config.metadata_csv).exists()
        for column in ("record_name", "signal_path", "stratify_class", "n_channels"):
            assert column in df.columns
        # Record ids and signal paths must survive as strings for wfdb.
        assert df["record_name"].iloc[0] == "300"
        assert df["signal_path"].iloc[0] == "300"

    def test_the_cached_csv_is_reread_with_record_ids_as_strings(self, tmp_path):
        """Second call takes the cache — and must not let pandas make "300" an int."""
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.stdb import STDBSplitter

        self._tree(tmp_path, self._records())
        config = load_config("stdb")
        splitter = STDBSplitter()
        splitter.load_metadata(tmp_path, config)
        again = splitter.load_metadata(tmp_path, config)

        # What matters is that the values stay strings — the exact pandas dtype
        # for a string column has changed across versions.
        assert all(isinstance(v, str) for v in again["record_name"])
        assert all(isinstance(v, str) for v in again["signal_path"])
        assert list(again["record_name"])[:3] == ["300", "301", "302"]

    def test_stratification_is_the_group_crossed_with_the_channel_count(self, tmp_path):
        """14 depression_2ch / 9 depression_1ch / 4 elevation_2ch / 1 elevation_1ch.

        Both axes matter and neither alone is enough: the group is the only thing
        the release documents, and the channel count is what decides whether a
        record can be batched with its fold-mates.
        """
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.stdb import STDBSplitter

        self._tree(tmp_path, self._records())
        config = load_config("stdb")
        splitter = STDBSplitter()
        df = splitter.load_metadata(tmp_path, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.value_counts().to_dict() == {
            "depression_2ch": 14,
            "depression_1ch": 9,
            "elevation_2ch": 4,
            "elevation_1ch": 1,
        }
        # The largest class clears 10 folds, which is what keeps StratifiedKFold
        # from raising despite the singleton.
        assert labels.value_counts().max() >= 10

    def test_the_five_long_term_excerpts_are_the_ones_the_release_names(self, tmp_path):
        """323-327 and nothing else — the assignment is by record name, not by data."""
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.stdb import STDBSplitter

        self._tree(tmp_path, self._records())
        df = STDBSplitter().load_metadata(tmp_path, load_config("stdb"))
        elevation = set(df.loc[df["st_change_type"] == "elevation", "record_name"])

        assert elevation == {"323", "324", "325", "326", "327"}
        assert set(df["group_source"]) == {"landing_page"}

    def test_missing_stratify_column_raises_rather_than_silently_restratifying(self):
        """A frame not produced by load_metadata must fail loudly."""
        import pandas as pd

        from ecgbench.config import load_config
        from ecgbench.splitting.strategies.stdb import STDBSplitter

        df = pd.DataFrame({"record_name": ["300"], "signal_path": ["300"]})
        with pytest.raises(ValueError, match="stratify_class"):
            STDBSplitter().get_stratification_labels(df, load_config("stdb"))

    def test_the_registry_resolves_stdb_to_its_own_splitter(self):
        """A slug mismatch would silently fall back to GenericSplitter."""
        from ecgbench.splitting import get_splitter
        from ecgbench.splitting.strategies.stdb import STDBSplitter

        assert isinstance(get_splitter("stdb"), STDBSplitter)

    def test_folds_keep_both_channel_layouts_in_every_fold(self, tmp_path):
        """The reason the channel count is crossed into the stratification at all.

        With 10 single-channel records over 10 folds, an unstratified split could
        leave a fold holding none — or three — and a user selecting
        ``leads=["ECG1","ECG2"]`` would then see fold sizes vary by a factor of
        three for a reason nothing announces.
        """
        pytest.importorskip("wfdb")

        from ecgbench.config import load_config
        from ecgbench.splitting.engine import split_dataset
        from ecgbench.splitting.strategies.stdb import STDBSplitter

        config = load_config("stdb")
        self._tree(tmp_path, self._records())
        splitter = STDBSplitter()
        df = splitter.load_metadata(tmp_path, config)
        labels = splitter.get_stratification_labels(df, config)
        result = split_dataset(df, labels, config, n_folds=10)

        channels = dict(zip(df["record_name"], df["n_channels"]))
        per_fold = {
            fold: {channels[r] for r in frame["record_name"]}
            for fold, frame in result.folds.items()
        }
        assert len(per_fold) == 10
        for fold, layouts in per_fold.items():
            assert layouts <= {1, 2}
            assert 2 in layouts, f"fold {fold} holds no two-channel record"
        # And all but one fold holds a single-channel record too, on this seed.
        assert sum(1 in layouts for layouts in per_fold.values()) >= 9


class TestApneaECGSplitter:
    """70 records, 30 subjects, and a predefined split that must not be used.

    Every other splitter here either has a patient column handed to it or has
    genuinely one record per patient. This one exists because Apnea-ECG ships no
    subject identifier at all while containing up to four nights per subject, so
    the grouping has to be reconstructed before a fold can be assigned.
    """

    INFO_HEADER = (
        "Additional information about the recordings used in the "
        "PhysioNet/CinC Challenge 2000\n"
        "\n"
        "Record\tLength\tnon-apn\tapnea\thours\tAI\tHI\tAHI\tAge\tSex\theight\tweight\n"
        "\n"
    )

    #: (record, apnea_minutes, total_minutes, age, sex, height, weight).
    #: a01/a14 share all four demographic fields, and so do a02/x01 — the latter
    #: pair straddling the challenge's learning/test boundary exactly as 18 real
    #: subjects do. 7 records, 5 subjects.
    RECORDS = (
        ("a01", 120, 140, 51, "M", 175, 102),
        ("a14", 110, 140, 51, "M", 175, 102),
        ("a02", 130, 140, 38, "M", 180, 120),
        ("b01", 20, 140, 44, "F", 170, 63),
        ("c01", 0, 140, 31, "M", 184, 74),
        ("x01", 125, 140, 38, "M", 180, 120),
        ("x30", 115, 140, 44, "M", 177, 105),
    )

    def _tree(self, tmp_path):
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        info = [self.INFO_HEADER]
        for rec, apnea, total, age, sex, height, weight in self.RECORDS:
            n_samples = total * 6000
            (tmp_path / f"{rec}.hea").write_text(
                f"{rec} 1 100 {n_samples}\n{rec}.dat 16 200 12 0 -12 5827 0 ECG\n",
                encoding="utf-8",
            )
            np.zeros(n_samples, dtype=np.int16).tofile(tmp_path / f"{rec}.dat")
            wfdb.wrann(
                rec, "apn",
                sample=np.arange(total, dtype=np.int64) * 6000,
                symbol=list("A" * apnea + "N" * (total - apnea)),
                fs=100,
                write_dir=str(tmp_path),
            )
            wfdb.wrann(
                rec, "qrs",
                sample=np.arange(1, total * 60) * 100,
                symbol=["N"] * (total * 60 - 1),
                fs=100,
                write_dir=str(tmp_path),
            )
            info.append(
                f"{rec}\t{total}\t{total - apnea}\t{apnea}\t9\t10\t5\t15.0\t"
                f"{age}\t{sex}\t{height}\t{weight}\t\t\n"
            )
        (tmp_path / "RECORDS").write_text(
            "\n".join(rec for rec, *_ in self.RECORDS) + "\n", encoding="utf-8"
        )
        (tmp_path / "additional-information.txt").write_text(
            "".join(info), encoding="utf-8"
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="apnea_ecg", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_name", patient_id_column="subject_id",
            signal_path_columns={100: "signal_path"}, default_sampling_rate=100,
            label_column="apnea_class", leads=1, zero_padded_identifiers=False,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.apnea_ecg import ApneaECGSplitter

        assert isinstance(get_splitter("apnea_ecg"), ApneaECGSplitter)

    def test_builds_metadata_with_subject_grouping_and_derived_class(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        tree = self._tree(tmp_path)
        config = self._config(sample_config)

        df = get_splitter("apnea_ecg").load_metadata(tree, config).set_index("record_name")

        assert len(df) == 7
        # Two subjects have two records each, so 7 records make 5 subjects.
        assert df["subject_id"].nunique() == 5
        assert df.loc["a01", "subject_id"] == df.loc["a14", "subject_id"]
        # The leak this dataset actually has: one subject on both sides of the
        # challenge's own learning/test division.
        assert df.loc["a02", "subject_id"] == df.loc["x01", "subject_id"]
        assert df.loc["a02", "challenge_set"] == "learning"
        assert df.loc["x01", "challenge_set"] == "test"
        assert df.loc["a01", "apnea_class"] == "A"
        assert df.loc["b01", "apnea_class"] == "B"
        assert df.loc["c01", "apnea_class"] == "C"
        assert df.loc["a01", "signal_path"] == "a01"
        assert set(df["challenge_set"]) == {"learning", "test"}

    def test_metadata_csv_is_written_to_disk_because_validation_re_reads_it(
        self, sample_config, tmp_path
    ):
        """validate_dataset reads the CSV itself; an in-memory frame is invisible.

        This is the trap that shipped Chapman broken for months — a path or column
        fix-up living only in load_metadata.
        """
        pytest.importorskip("wfdb")
        tree = self._tree(tmp_path)
        config = self._config(sample_config)

        assert not (tree / config.metadata_csv).exists()
        get_splitter("apnea_ecg").load_metadata(tree, config)
        assert (tree / config.metadata_csv).exists()

        # Second call reads the cache and agrees with the first.
        cached = get_splitter("apnea_ecg").load_metadata(tree, config)
        assert len(cached) == 7
        assert cached["subject_id"].nunique() == 5

    def test_stratification_labels_come_from_the_label_loader_s_column(
        self, sample_config, tmp_path
    ):
        pytest.importorskip("wfdb")
        tree = self._tree(tmp_path)
        config = self._config(sample_config)
        splitter = get_splitter("apnea_ecg")

        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "apnea_class"
        assert labels.value_counts().to_dict() == {"A": 5, "B": 1, "C": 1}

    def test_stratification_refuses_a_frame_the_loader_did_not_produce(
        self, sample_config
    ):
        """Recomputing the mapping here is what let PTB-XL's two copies drift."""
        import pandas as pd

        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("apnea_ecg").get_stratification_labels(
                pd.DataFrame({"record_name": ["a01"]}), self._config(sample_config)
            )

    def test_grouped_split_keeps_every_subject_in_one_fold(
        self, sample_config, tmp_path
    ):
        """The point of the whole exercise, end to end through split_dataset.

        With 5 subject groups and 7 records, an ungrouped split would be free to
        put a01 in train and a14 in test — the failure mode the release's own
        learning/test division actually has, for 49 of its 70 records.
        """
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        tree = self._tree(tmp_path)
        config = self._config(sample_config)
        splitter = get_splitter("apnea_ecg")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        result = split_dataset(df, labels, config, n_folds=5)

        assert result.group_column == "subject_id"
        assert set(result.folds) == {1, 2, 3, 4, 5}

        assigned = pd.concat(
            [frame.assign(fold=fold) for fold, frame in result.folds.items()],
            ignore_index=True,
        )
        assert len(assigned) == 7
        folds_per_subject = assigned.groupby("subject_id")["fold"].nunique()
        assert (folds_per_subject == 1).all()
        # Specifically: the pair that straddles the challenge's own split does
        # not straddle an ECGBench fold.
        a02_fold = assigned.loc[assigned["record_name"] == "a02", "fold"].iloc[0]
        x01_fold = assigned.loc[assigned["record_name"] == "x01", "fold"].iloc[0]
        assert a02_fold == x01_fold


class TestECGIDDBSplitter:
    """310 records, 90 subjects, and a label that is also the grouping column.

    ECG-ID is the one dataset here whose ground truth is identity, so the split
    that protects against leakage is exactly the split that makes its own task
    impossible. That is a documented consequence, not a bug, and these tests pin
    both halves: subjects do not span folds, and the reason it matters is stated.
    """

    #: (subject, rec, age, sex, date). 12 records from 5 subjects: Person_01 has
    #: four across two dates, Person_02 three on one date, and the rest one each —
    #: the release's own shape in miniature.
    RECORDS = (
        ("Person_01", "rec_1", 25, "male", "07.12.2004"),
        ("Person_01", "rec_2", 25, "male", "07.12.2004"),
        ("Person_01", "rec_3", 25, "male", "28.12.2004"),
        ("Person_01", "rec_4", 25, "male", "28.12.2004"),
        ("Person_02", "rec_1", 22, "female", "12.05.2005"),
        ("Person_02", "rec_2", 22, "female", "12.05.2005"),
        ("Person_02", "rec_3", 22, "female", "12.05.2005"),
        ("Person_03", "rec_1", 45, "female", "12.05.2005"),
        ("Person_03", "rec_2", 45, "female", "12.05.2005"),
        ("Person_04", "rec_1", 58, "male", "12.05.2005"),
        ("Person_05", "rec_1", 19, "male", "12.05.2005"),
        ("Person_05", "rec_2", 19, "male", "12.05.2005"),
    )

    HEADER = (
        "{rec} 2 500 10000\n"
        "{rec}.dat 16 200 12 0 -17 17532 0 ECG I\n"
        "{rec}.dat 16 200 12 0 -23 2004 0 ECG I filtered\n"
        "\n"
        "# Age: {age}\n"
        "# Sex: {sex}\n"
        "# ECG date: {date}\n"
    )

    def _tree(self, tmp_path):
        wfdb = pytest.importorskip("wfdb")
        import numpy as np

        for subject, rec, age, sex, date in self.RECORDS:
            directory = tmp_path / subject
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{rec}.hea").write_text(
                self.HEADER.format(rec=rec, age=age, sex=sex, date=date), encoding="utf-8"
            )
            np.zeros(10000 * 2, dtype=np.int16).tofile(directory / f"{rec}.dat")
            r_peaks = 200 + np.arange(10) * 400
            sample = np.empty(20, dtype=np.int64)
            sample[0::2] = r_peaks
            sample[1::2] = r_peaks + 100
            wfdb.wrann(
                rec, "atr",
                sample=sample,
                symbol=["N", "t"] * 10,
                fs=500,
                write_dir=str(directory),
            )
        (tmp_path / "RECORDS").write_text(
            "\n".join(f"{subject}/{rec}" for subject, rec, *_ in self.RECORDS) + "\n",
            encoding="utf-8",
        )
        return tmp_path

    def _config(self, sample_config):
        from dataclasses import replace

        return replace(
            sample_config, slug="ecgiddb", metadata_csv="ecgbench_metadata.csv",
            record_id_column="record_id", patient_id_column="subject_id",
            signal_path_columns={500: "signal_path"}, default_sampling_rate=500,
            label_column="subject_id", leads=2, zero_padded_identifiers=False,
        )

    def test_registered_splitter_is_not_the_generic_fallback(self):
        from ecgbench.splitting.strategies.ecgiddb import ECGIDDBSplitter

        assert isinstance(get_splitter("ecgiddb"), ECGIDDBSplitter)

    def test_builds_metadata_with_disambiguated_ids_and_release_paths(
        self, sample_config, tmp_path
    ):
        """``rec_1`` names five recordings here and 90 in the release."""
        pytest.importorskip("wfdb")
        tree = self._tree(tmp_path)
        config = self._config(sample_config)

        df = get_splitter("ecgiddb").load_metadata(tree, config)

        assert len(df) == 12
        assert df["record_id"].is_unique
        assert df["subject_id"].nunique() == 5
        # The bare record name is NOT a key — five records are called rec_1.
        assert (df["record_name"] == "rec_1").sum() == 5
        row = df.set_index("record_id").loc["Person_01_rec_1"]
        assert row["subject_id"] == "Person_01"
        # The path wfdb is handed keeps the release's own subdirectory form.
        assert row["signal_path"] == "Person_01/rec_1"
        assert row["stratify_class"] == "male_le30"
        assert row["n_records_for_subject"] == 4
        assert row["n_sessions_for_subject"] == 2

    def test_metadata_csv_is_written_to_disk_because_validation_re_reads_it(
        self, sample_config, tmp_path
    ):
        """validate_dataset reads the CSV itself; an in-memory frame is invisible.

        And the round trip has to preserve the identifiers: ``Person_01`` is
        zero-padded, so it survives only because the prefix keeps it from being an
        all-digits column.
        """
        pytest.importorskip("wfdb")
        tree = self._tree(tmp_path)
        config = self._config(sample_config)
        splitter = get_splitter("ecgiddb")

        first = splitter.load_metadata(tree, config)
        written = tree / "ecgbench_metadata.csv"
        assert written.exists()

        # Second call reads the cache rather than rescanning 310 headers.
        cached = splitter.load_metadata(tree, config)
        assert list(cached["record_id"]) == list(first["record_id"])
        assert list(cached["subject_id"]) == list(first["subject_id"])
        assert cached["subject_id"].iloc[0] == "Person_01"  # not 1, and not "1"
        assert cached["signal_path"].iloc[0] == "Person_01/rec_1"

    def test_stratification_labels_come_from_the_loader(self, sample_config, tmp_path):
        pytest.importorskip("wfdb")
        tree = self._tree(tmp_path)
        config = self._config(sample_config)
        splitter = get_splitter("ecgiddb")

        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        assert labels.name == "sex_x_age"
        assert list(labels) == list(df["stratify_class"])
        assert set(labels) == {"male_le30", "female_le30", "female_gt30", "male_gt30"}

    def test_stratification_refuses_a_frame_it_did_not_build(self, sample_config):
        """``stratify_class`` comes from the label loader, never from a re-derivation."""
        config = self._config(sample_config)
        with pytest.raises(ValueError, match="stratify_class"):
            get_splitter("ecgiddb").get_stratification_labels(
                pd.DataFrame({"record_id": ["Person_01_rec_1"], "age": [25]}), config
            )

    def test_no_subject_spans_a_fold(self, sample_config, tmp_path):
        """The whole point of the grouping — and the reason the folds cannot be

        used for identification. A model trained on Person_01's other three records
        and tested on the fourth would be measuring nothing; a model that never sees
        Person_01 cannot recognise them. Both statements are true here, which is why
        the label loader points at a within-subject session split instead.
        """
        pytest.importorskip("wfdb")
        from ecgbench.splitting import split_dataset

        tree = self._tree(tmp_path)
        config = self._config(sample_config)
        splitter = get_splitter("ecgiddb")
        df = splitter.load_metadata(tree, config)
        labels = splitter.get_stratification_labels(df, config)

        # 5 subjects, so 5 folds is the most that can be filled at all.
        result = split_dataset(df, labels, config, n_folds=5)

        assigned = pd.concat(
            [frame.assign(fold=fold) for fold, frame in result.folds.items()],
            ignore_index=True,
        )
        assert len(assigned) == 12
        folds_per_subject = assigned.groupby("subject_id")["fold"].nunique()
        assert (folds_per_subject == 1).all()
        # Concretely: all four of Person_01's records land together.
        person_01 = assigned.loc[assigned["subject_id"] == "Person_01", "fold"]
        assert len(person_01) == 4
        assert person_01.nunique() == 1
        assert result.group_column == "subject_id"
