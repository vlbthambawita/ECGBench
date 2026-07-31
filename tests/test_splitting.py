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
