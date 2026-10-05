"""Record-level access (metadata layer, Phase 6): ``load_records``, ``query_records``,
the factored fold-table readers ``ECGDataset`` now shares, and ``ecgbench records``.

Fixtures come from conftest: ``tmp_wfdb_signal_dataset`` is five records with a
five-fold tree (folds 1-3 train, 4 val, 5 test, one record per fold) and
``sample_config`` describes it. Nothing touches the network: the Hub path is
exercised by pointing ``hf_hub_download`` at the local tree.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pandas as pd
import pytest

from ecgbench.cli import main
from ecgbench.config import LabelConfig, load_config
from ecgbench.metadata.identity import UnknownDatasetError
from ecgbench.metadata.records import (
    HF_REPO_ID,
    RecordsQueryError,
    RecordsUnavailableError,
    SplitsNotPublishedError,
    fetch_hub_fold_table,
    filter_fold_table,
    hub_fold_paths,
    hub_fold_url,
    hub_view_sql,
    join_labels,
    load_fold_table,
    load_records,
    query_records,
    resolve_config,
)

FOLD_COLUMNS = ["record_id", "filename", "fold", "default_split"]


@pytest.fixture
def labelled(tmp_wfdb_signal_dataset, sample_config):
    """The WFDB fixture plus a declarative label CSV keyed by ``record_id``.

    ``filename`` is deliberately present in the label CSV as well, with bogus
    values, so the join's "fold-table columns win" rule is exercised; ``rec_9``
    has a label row but no record, which a left join must ignore.
    """
    root = tmp_wfdb_signal_dataset
    pd.DataFrame(
        {
            "record_id": [f"rec_{i}" for i in range(5)] + ["rec_9"],
            "filename": ["bogus"] * 6,
            "diagnosis": ["AFIB", "NORM", "AFIB", "NORM", "STTC", "NORM"],
            "age": [61, 44, 78, 30, 55, 20],
        }
    ).to_csv(root / "labels.csv", index=False)
    config = replace(
        sample_config,
        labels=LabelConfig(source_csv="labels.csv", join_column="record_id"),
    )
    return config, root


@pytest.fixture
def fake_hub(labelled, monkeypatch):
    """``hf_hub_download`` resolving ``<slug>/<version>/...`` against the local tree."""
    config, root = labelled
    calls: list[str] = []

    def download(repo_id, filename, repo_type):
        assert repo_id == HF_REPO_ID and repo_type == "dataset"
        assert filename.startswith(config.slug + "/")
        calls.append(filename)
        local = root / filename.split("/", 1)[1]
        if not local.is_file():
            raise FileNotFoundError(filename)
        return str(local)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


# --------------------------------------------------------------------------- fold tables


class TestFoldTables:
    def test_local_tree_every_record(self, labelled):
        config, root = labelled
        df = load_fold_table(config, "clean", None, None, source="local", splits_dir=root)
        assert list(df.columns) == FOLD_COLUMNS
        assert sorted(df["record_id"]) == [f"rec_{i}" for i in range(5)]
        assert sorted(df["fold"]) == [1, 2, 3, 4, 5]

    def test_split_reads_its_fold_files(self, labelled):
        config, root = labelled
        df = load_fold_table(config, "clean", "train", None, source="local", splits_dir=root)
        assert sorted(df["fold"]) == [1, 2, 3]
        assert set(df["default_split"]) == {"train"}

    def test_fold_outside_the_split_names_the_fix(self, labelled):
        config, root = labelled
        with pytest.raises(FileNotFoundError, match="split=None"):
            load_fold_table(config, "clean", "train", [4], source="local", splits_dir=root)

    def test_split_none_crosses_split_boundaries(self, labelled):
        config, root = labelled
        df = load_fold_table(config, "clean", None, [1, 5], source="local", splits_dir=root)
        assert sorted(df["default_split"]) == ["test", "train"]

    def test_unknown_fold_lists_the_available_ones(self):
        master = pd.DataFrame({"fold": [1, 2], "default_split": ["train", "train"]})
        with pytest.raises(ValueError, match=r"Fold\(s\) \[7\].*Available: \[1, 2\]"):
            filter_fold_table(master, None, [7])

    def test_bad_version_split_and_source_are_rejected(self, labelled):
        config, root = labelled
        with pytest.raises(ValueError, match="version"):
            load_fold_table(config, "dirty", None, None, source="local", splits_dir=root)
        with pytest.raises(ValueError, match="split"):
            load_fold_table(config, "clean", "dev", None, source="local", splits_dir=root)
        with pytest.raises(ValueError, match="source"):
            load_fold_table(config, "clean", None, None, source="ftp", splits_dir=root)
        with pytest.raises(ValueError, match="splits_dir"):
            load_fold_table(config, "clean", None, None, source="local")

    def test_hub_paths_follow_the_repo_layout(self, sample_config):
        c = sample_config
        assert hub_fold_paths(c, "clean", None, None) == ["test_dataset/clean/folds.csv"]
        assert hub_fold_paths(c, "original", "val", None) == ["test_dataset/original/folds.csv"]
        assert hub_fold_paths(c, "clean", None, [2, 9]) == ["test_dataset/clean/folds.csv"]
        assert hub_fold_paths(c, "clean", "train", [1, 2]) == [
            "test_dataset/clean/train/fold_1.csv",
            "test_dataset/clean/train/fold_2.csv",
        ]

    def test_hub_fetch_per_fold_files_and_master(self, labelled, fake_hub):
        config, _ = labelled
        df = fetch_hub_fold_table(config, "clean", "train", [1, 2])
        assert fake_hub == [
            "test_dataset/clean/train/fold_1.csv",
            "test_dataset/clean/train/fold_2.csv",
        ]
        assert sorted(df["fold"]) == [1, 2]

        fake_hub.clear()
        df = fetch_hub_fold_table(config, "clean", "val", None)
        assert fake_hub == ["test_dataset/clean/folds.csv"]
        assert df["fold"].tolist() == [4]

    def test_withheld_dataset_raises_before_any_download(self, monkeypatch):
        import huggingface_hub

        def boom(**kwargs):
            raise AssertionError("must not download")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
        config = load_config("mimic_iv_ecg")
        assert not config.publish_fold_csvs
        with pytest.raises(SplitsNotPublishedError, match="ecgbench splits"):
            fetch_hub_fold_table(config, "clean", "train", None)
        with pytest.raises(SplitsNotPublishedError):
            hub_fold_url(config)

    def test_ecgdataset_hub_path_goes_through_the_shared_fetch(self, labelled, fake_hub):
        """The HF path of ECGDataset was untested before it was factored out."""
        pytest.importorskip("torch")
        from ecgbench.dataset import ECGDataset

        config, root = labelled
        ds = ECGDataset(config, split="train", version="clean", data_path=root)
        assert ds.metadata_source == "hf"
        assert len(ds) == 3 and fake_hub == ["test_dataset/clean/folds.csv"]
        assert tuple(ds[0]["signal"].shape) == (12, 5000)

    def test_hub_view_sql_declares_text_ids_only_when_padded(self, sample_config):
        plain = hub_view_sql(sample_config, "hf://x/folds.csv")
        assert plain == (
            "CREATE VIEW records AS SELECT * FROM read_csv('hf://x/folds.csv', header = true)"
        )
        padded = hub_view_sql(load_config("afdb"), "hf://x/folds.csv")
        assert "types = {'record_name': 'VARCHAR', 'signal_path': 'VARCHAR'}" in padded
        # the statement is DuckDB-parsable: a view over a local file stands in for hf://
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect()
        con.execute("SET enable_external_access = false")
        for statement in (plain, padded):
            with pytest.raises(duckdb.Error) as exc:
                con.execute(statement)
            assert "Parser" not in type(exc.value).__name__  # it fails on access, not syntax

    def test_hub_url(self, sample_config):
        assert hub_fold_url(sample_config) == (
            "hf://datasets/vlbthambawita/ECGBench/test_dataset/clean/folds.csv"
        )
        with pytest.raises(ValueError, match="version"):
            hub_fold_url(sample_config, "dirty")


# --------------------------------------------------------------------------- records


class TestLoadRecords:
    def test_joins_labels_fold_columns_first(self, labelled):
        config, root = labelled
        df = load_records(config, data_path=root, source="local")
        assert list(df.columns) == FOLD_COLUMNS + ["diagnosis", "age"]
        assert len(df) == 5
        by_id = df.set_index("record_id")
        assert by_id.loc["rec_2", "diagnosis"] == "AFIB" and by_id.loc["rec_2", "age"] == 78
        # the label CSV's clashing filename column is dropped, not suffixed
        assert by_id.loc["rec_2", "filename"] == "records/rec_2"
        assert "filename_x" not in df.columns and "filename_y" not in df.columns

    def test_split_fold_and_version_pass_through(self, labelled):
        config, root = labelled
        val = load_records(config, data_path=root, source="local", split="val")
        assert val["record_id"].tolist() == ["rec_3"] and val["diagnosis"].tolist() == ["NORM"]
        two = load_records(config, data_path=root, source="local", fold_numbers=[1, 5])
        assert sorted(two["fold"]) == [1, 5]
        original = load_records(config, data_path=root, source="local", version="original")
        assert len(original) == 5

    def test_labels_false_is_the_fold_table(self, labelled):
        config, root = labelled
        df = load_records(config, data_path=root, source="local", labels=False)
        assert list(df.columns) == FOLD_COLUMNS

    def test_splits_dir_separate_from_data_path(self, labelled, tmp_path):
        config, root = labelled
        out = tmp_path / "output" / config.slug
        shutil.copytree(root / "clean", out / "clean")
        df = load_records(config, data_path=root, splits_dir=out, source="local", split="test")
        assert df["record_id"].tolist() == ["rec_4"] and df["diagnosis"].tolist() == ["STTC"]

    def test_hub_source_with_labels(self, labelled, fake_hub):
        config, root = labelled
        df = load_records(config, data_path=root, split="val")
        assert fake_hub == ["test_dataset/clean/folds.csv"]
        assert df["diagnosis"].tolist() == ["NORM"]

    def test_missing_label_rows_warn_but_keep_the_record(self, labelled, caplog):
        config, root = labelled
        labels = pd.read_csv(root / "labels.csv")
        labels[labels["record_id"] != "rec_1"].to_csv(root / "labels.csv", index=False)
        with caplog.at_level("WARNING", logger="ecgbench.metadata.records"):
            df = load_records(config, data_path=root, source="local")
        assert len(df) == 5
        assert df.set_index("record_id").loc["rec_1"].isna()[["diagnosis", "age"]].all()
        assert "1 of 5 records have no label row" in caplog.text

    def test_no_match_at_all_names_the_join_columns(self, labelled):
        config, root = labelled
        fold = pd.DataFrame({"record_id": ["a", "b"], "fold": [1, 2]})
        labels = pd.DataFrame({"x": [1]}, index=pd.Index(["zzz"], name="record_id"))
        with pytest.raises(ValueError, match="labels.csv.*record_id"):
            join_labels(fold, labels, config)

    def test_int_and_str_ids_still_join(self, sample_config):
        fold = pd.DataFrame({"record_id": ["1", "2"], "fold": [1, 2]})
        labels = pd.DataFrame({"x": [10, 20]}, index=pd.Index([1, 2], name="record_id"))
        df = join_labels(fold, labels, sample_config)
        assert df["x"].tolist() == [10, 20]
        assert df["record_id"].tolist() == ["1", "2"]  # the fold table's column is untouched

    def test_labels_unavailable_dataset(self, tmp_path):
        from ecgbench.labels import LabelsUnavailableError

        config = load_config("mimic_iv_ecg_demo")
        (tmp_path / "clean").mkdir()
        pd.DataFrame(
            {config.record_id_column: ["1"], "fold": [1], "default_split": ["train"]}
        ).to_csv(tmp_path / "clean" / "folds.csv", index=False)
        with pytest.raises(LabelsUnavailableError):
            load_records(config, data_path=tmp_path, source="local")
        assert len(load_records(config, data_path=tmp_path, source="local", labels=False)) == 1


class TestResolveConfig:
    def test_any_alias_resolves(self):
        assert resolve_config("PTB-XL").slug == "ptbxl"
        assert resolve_config("mit-bih-arrhythmia-database").slug == "mitdb"

    def test_catalogue_only_and_unknown(self):
        with pytest.raises(RecordsUnavailableError, match="catalogue-only"):
            resolve_config("ptb-xl-plus")
        with pytest.raises(UnknownDatasetError):
            resolve_config("no-such-dataset")


# --------------------------------------------------------------------------- SQL


class TestQueryRecords:
    def test_sql_over_the_records_view(self, labelled):
        pytest.importorskip("duckdb")
        config, root = labelled
        df = load_records(config, data_path=root, source="local")
        out = query_records(
            df, "select default_split, count(*) as n from records group by 1 order by 1"
        )
        assert out.values.tolist() == [["test", 1], ["train", 3], ["val", 1]]

    def test_bad_sql_is_a_query_error(self):
        pytest.importorskip("duckdb")
        with pytest.raises(RecordsQueryError, match="nope"):
            query_records(pd.DataFrame({"a": [1]}), "select * from nope")

    def test_the_file_system_is_out_of_reach(self, tmp_path):
        pytest.importorskip("duckdb")
        secret = tmp_path / "secret.csv"
        secret.write_text("a\n1\n")
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(RecordsQueryError, match="file system operations are disabled"):
            query_records(df, f"select * from read_csv('{secret}')")
        with pytest.raises(RecordsQueryError):
            query_records(df, "SET enable_external_access = true")
        with pytest.raises(RecordsQueryError):
            query_records(df, f"COPY records TO '{tmp_path / 'out.parquet'}'")

    def test_without_duckdb_the_hint_names_the_extra(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "duckdb", None)
        with pytest.raises(ImportError, match=r"ecgbench\[analytics\]"):
            query_records(pd.DataFrame({"a": [1]}), "select 1")


# --------------------------------------------------------------------------- CLI


class TestCli:
    @pytest.fixture
    def cli_dataset(self, labelled, monkeypatch):
        """Make ``ecgbench records test_dataset`` resolve to the fixture config."""
        config, root = labelled
        import ecgbench.metadata.records as rec

        monkeypatch.setattr(rec, "resolve_config", lambda key: config)
        return config, root

    def _run(self, capsys, *argv):
        code = main(["records", *argv])
        out, err = capsys.readouterr()
        return code, out, err

    def test_csv_output(self, cli_dataset, capsys):
        config, root = cli_dataset
        code, out, _ = self._run(
            capsys, "test_dataset", "--data-path", str(root), "--source", "local",
            "--split", "val", "--format", "csv",
        )
        assert code == 0
        assert out.splitlines() == ["record_id,filename,fold,default_split,diagnosis,age",
                                    "rec_3,records/rec_3,4,val,NORM,30"]

    def test_json_output(self, cli_dataset, capsys):
        config, root = cli_dataset
        code, out, _ = self._run(
            capsys, "test_dataset", "--splits-dir", str(root), "--data-path", str(root),
            "--fold", "5", "--format", "json",
        )
        assert code == 0
        assert json.loads(out) == [
            {"record_id": "rec_4", "filename": "records/rec_4", "fold": 5,
             "default_split": "test", "diagnosis": "STTC", "age": 55}
        ]

    def test_table_output_is_limited_and_says_so(self, cli_dataset, capsys):
        config, root = cli_dataset
        code, out, _ = self._run(
            capsys, "test_dataset", "--splits-dir", str(root), "--no-labels", "--limit", "2"
        )
        assert code == 0
        lines = out.rstrip("\n").splitlines()
        assert lines[0].split() == FOLD_COLUMNS
        assert len(lines) == 5 and lines[-1] == "(2 of 5 rows; --limit 0 shows all)"
        code, out, _ = self._run(
            capsys, "test_dataset", "--splits-dir", str(root), "--no-labels", "--limit", "0"
        )
        assert out.count("rec_") == 2 * 5  # record_id and filename columns, every row

    def test_sql_output(self, cli_dataset, capsys):
        pytest.importorskip("duckdb")
        config, root = cli_dataset
        code, out, _ = self._run(
            capsys, "test_dataset", "--splits-dir", str(root), "--data-path", str(root),
            "--sql", "select diagnosis, count(*) as n from records group by 1 order by 1",
            "--format", "csv",
        )
        assert code == 0
        assert out.splitlines() == ["diagnosis,n", "AFIB,2", "NORM,2", "STTC,1"]

    def test_parquet_output(self, cli_dataset, capsys, tmp_path):
        pytest.importorskip("duckdb")
        config, root = cli_dataset
        target = tmp_path / "out" / "records.parquet"
        code, out, _ = self._run(
            capsys, "test_dataset", "--splits-dir", str(root), "--no-labels",
            "--format", "parquet", "--output", str(target),
        )
        assert code == 0 and target.is_file() and "wrote 5 rows" in out
        import duckdb

        back = duckdb.connect().execute(f"select count(*) from '{target}'").fetchone()[0]
        assert back == 5
        code, _, err = self._run(capsys, "test_dataset", "--format", "parquet")
        assert code == 2 and "--output" in err

    def test_errors_exit_1_with_a_message(self, cli_dataset, capsys, monkeypatch):
        config, root = cli_dataset
        code, _, err = self._run(capsys, "test_dataset", "--splits-dir", str(root / "nowhere"))
        assert code == 1 and "Could not find fold CSVs" in err
        code, _, err = self._run(
            capsys, "test_dataset", "--splits-dir", str(root), "--split", "train", "--fold", "5"
        )
        assert code == 1 and "split=None" in err
        pytest.importorskip("duckdb")
        code, _, err = self._run(
            capsys, "test_dataset", "--splits-dir", str(root), "--data-path", str(root),
            "--sql", "select * from nowhere",
        )
        assert code == 1 and "nowhere" in err

    def test_unknown_and_catalogue_only_datasets(self, capsys):
        code, _, err = self._run(capsys, "no-such-dataset", "--no-labels")
        assert code == 1 and "no-such-dataset" in err
        code, _, err = self._run(capsys, "ptb-xl-plus", "--no-labels")
        assert code == 1 and "catalogue-only" in err

    def test_withheld_dataset_quotes_the_recipe(self, capsys):
        code, _, err = self._run(capsys, "mimic_iv_ecg", "--no-labels")
        assert code == 1 and "ecgbench splits" in err

    def test_labels_unavailable_suggests_no_labels(self, capsys, tmp_path):
        config = load_config("mimic_iv_ecg_demo")
        (tmp_path / "clean").mkdir()
        pd.DataFrame(
            {config.record_id_column: ["1"], "fold": [1], "default_split": ["train"]}
        ).to_csv(tmp_path / "clean" / "folds.csv", index=False)
        code, _, err = self._run(capsys, "mimic_iv_ecg_demo", "--splits-dir", str(tmp_path))
        assert code == 1 and "--no-labels" in err
