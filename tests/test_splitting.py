"""Tests for the splitting framework."""

from pathlib import Path

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
