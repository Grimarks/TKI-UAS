"""Tes skrip Kaggle tanpa GPU/jaringan: paritas kontrak, shard/merge/resume, pasangan data, pembuat kernel."""
import importlib.util
import json
import py_compile
import subprocess
import sys

import pytest

from hybrid_ir import keyphrases as kp
from hybrid_ir import text as txt


def _load(root, name):
    spec = importlib.util.spec_from_file_location(name, root / "kaggle" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen(root):
    return _load(root, "generate_keyphrases")


@pytest.fixture(scope="module")
def train(root):
    return _load(root, "train_encoder")


@pytest.fixture(scope="module")
def build_kernel(root):
    return _load(root, "build_kernel")


SAMPLES = ["", "  Hello   World ", "ﬁnal\x00 text", "Title?", "Neural networks. Deep learning!", "judul\ttab\nbaris"]


@pytest.mark.parametrize("s", SAMPLES)
def test_contract_normalize_and_fingerprint(gen, s):
    assert gen.normalize_text(s) == txt.normalize_text(s)
    assert gen.text_fingerprint(s) == txt.text_fingerprint(s)


@pytest.mark.parametrize("title,body", [("T", "B"), ("T?", "B"), ("", "B"), ("T", ""), (" a  b ", "c\nd")])
def test_contract_full_text(gen, title, body):
    assert gen.make_full_text(title, body) == txt.make_full_text(title, body)


@pytest.mark.parametrize("raw", ["a;b;A;c;;" + "w " * 20, "x;y;z", "", " ; ;"])
def test_contract_parse_and_weights(gen, raw):
    assert gen.parse_generated_keyphrases(raw, 3) == kp.parse_generated_keyphrases(raw, 3)
    assert gen.rank_weights(5) == kp.rank_weights(5)


def test_all_scripts_compile(root, tmp_path):
    for f in ("generate_keyphrases.py", "train_encoder.py", "build_kernel.py"):
        py_compile.compile(str(root / "kaggle" / f), cfile=str(tmp_path / (f + "c")), doraise=True)


def test_read_items_matches_local_pipeline_keys(gen, root):
    from hybrid_ir.data import load_beir_dir
    items = gen.read_items(root / "data" / "demo")
    keys = {it["key"] for it in items}
    ds = load_beir_dir(root / "data" / "demo")
    local = {txt.text_fingerprint(d.full_text) for d in ds.documents} | {txt.text_fingerprint(q) for q in ds.queries.values()}
    assert keys == local
    assert [it["key"] for it in items] == sorted(keys)


def test_worker_shard_merge_resume_and_oom(gen, tmp_path):
    items = [{"key": f"{i:040x}", "title": f"title {i}", "body": "x" * i} for i in range(1, 21)]
    calls = {"n": 0}

    def fake(sources):
        calls["n"] += 1
        if calls["n"] == 1 and len(sources) > 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return [f"alpha {s[:3]}; beta; alpha {s[:3]}" for s in sources]

    for shard in range(2):
        gen.run_worker(items[shard::2], fake, tmp_path / f"part-{shard}.jsonl", batch_size=8, top_n=5, log=lambda m: None)
    assert calls["n"] > 3                                 # ada percobaan ulang setelah OOM
    n = gen.merge_parts(tmp_path, 2, tmp_path / "out.jsonl", [it["key"] for it in items])
    assert n == 20
    rows = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["key"] for r in rows] == sorted(r["key"] for r in rows)
    assert all(len(r["keyphrases"]) == 2 for r in rows)   # duplikat 'alpha' dibuang

    before = calls["n"]                                   # resume: tidak ada yang tersisa
    gen.run_worker(items[0::2], fake, tmp_path / "part-0.jsonl", batch_size=8, top_n=5, log=lambda m: None)
    assert calls["n"] == before


def test_merge_fails_when_incomplete(gen, tmp_path):
    (tmp_path / "part-0.jsonl").write_text(json.dumps({"key": "a" * 40, "keyphrases": []}) + "\n")
    with pytest.raises(RuntimeError, match="belum punya"):
        gen.merge_parts(tmp_path, 1, tmp_path / "o.jsonl", ["a" * 40, "b" * 40])


def test_output_is_readable_by_precomputed_backend(gen, tmp_path):
    from hybrid_ir.keyphrases import PrecomputedKeyphrases, TextItem
    item = TextItem("A title", "Some abstract body")
    key = gen.text_fingerprint(gen.make_full_text(item.title, item.body))
    gen.run_worker([{"key": key, "title": item.title, "body": item.body}], lambda s: ["foo bar; baz"], tmp_path / "p.jsonl", 4, 5, log=lambda m: None)
    got = PrecomputedKeyphrases([tmp_path / "p.jsonl"]).extract([item], 5)
    assert [k for k, _ in got[0]] == ["foo bar", "baz"]


def test_kp20k_pairs_modes_and_filters(train):
    recs = [{"title": f"A long enough title {i}", "abstract": "word " * 60, "keyphrases": ["kp a", "kp b"]} for i in range(10)]
    recs.append({"title": "short", "abstract": "tiny", "keyphrases": []})
    assert len(train.kp20k_pairs(recs, "kp20k_title", 100, 1)) == 10
    assert len(train.kp20k_pairs(recs, "kp20k_mixed", 100, 1)) == 20
    kp = train.kp20k_pairs(recs, "kp20k_keyphrase", 100, 1)
    assert all(a == "kp a; kp b" and p.startswith("A long enough title") for a, p in kp)
    assert len(train.kp20k_pairs(recs, "kp20k_mixed", 5, 1)) == 5
    assert train.kp20k_pairs(recs, "kp20k_mixed", 100, 7) == train.kp20k_pairs(recs, "kp20k_mixed", 100, 7)


def test_json_reader_array_and_lines(train, tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"x": 1}, {"x": 2}]))
    (tmp_path / "b.jsonl").write_text('{"x": 1}\n\n{"x": 2}\n')
    assert list(train.iter_json_records(tmp_path / "a.json")) == list(train.iter_json_records(tmp_path / "b.jsonl"))


def test_beir_train_pairs(train, tmp_path, root):
    d = tmp_path / "ds"
    (d / "qrels").mkdir(parents=True)
    for f in ("corpus.jsonl", "queries.jsonl"):
        (d / f).write_text((root / "data" / "demo" / f).read_text())
    (d / "qrels" / "train.tsv").write_text((root / "data" / "demo" / "qrels" / "test.tsv").read_text())
    pairs = train.beir_train_pairs(d, "train", 1000, 1)
    assert len(pairs) == 24 and all(isinstance(a, str) and isinstance(p, str) for a, p in pairs)


def test_build_kernel_output(build_kernel, tmp_path):
    folder = build_kernel.build("generate", "alice", "hybrid-ir-kp", ["alice/hybrid-ir-scifact"], True,
                                ["--data-dir", "/kaggle/input/x", "--require-gpus", "2"], tmp_path)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "alice/hybrid-ir-kp" and meta["title"] == "hybrid-ir-kp"
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"] and meta["is_private"]
    script = (folder / "script.py").read_text()
    compile(script, "script.py", "exec")                 # sintaks valid; __future__ tetap di paling atas
    assert script.startswith("from __future__ import annotations")
    assert "--require-gpus" in script and "len(sys.argv) == 1" in script


@pytest.mark.parametrize("kwargs", [
    {"username": "A/lice", "slug": "ok-slug"}, {"username": "alice", "slug": "Bad Slug"},
    {"username": "alice", "slug": "ok-slug", "dataset_sources": ["no-slash"]},
])
def test_build_kernel_rejects_bad_input(build_kernel, tmp_path, kwargs):
    base = dict(job="generate", username="alice", slug="ok-slug", dataset_sources=[], internet=True, script_args=[], out_root=tmp_path)
    base.update(kwargs)
    with pytest.raises(ValueError):
        build_kernel.build(**base)


def test_generated_script_runs_help_in_subprocess(build_kernel, tmp_path):
    folder = build_kernel.build("generate", "alice", "hybrid-ir-kp", [], True, ["--help"], tmp_path)
    r = subprocess.run([sys.executable, str(folder / "script.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--data-dir" in r.stdout


def test_precomputed_end_to_end_with_fallback(gen, demo_cfg, demo_dataset, tmp_path, fake_seq2seq):
    """Simulasi alur Kaggle -> lokal: berkas precomputed (palsu) dipakai untuk dokumen & query dataset;
    query bebas jatuh ke generator cadangan (seq2seq; di tes diganti generator tiruan)."""
    from hybrid_ir.retriever import HybridRetriever

    items = gen.read_items(demo_cfg.data.path)

    def fake(sources):
        # "model" palsu: ambil 3 kata pertama judul sebagai keyphrase
        return [";".join(s.split("<s>")[0].lower().split()[:3]) + ";shared topic" for s in sources]

    gen.run_worker(items, fake, tmp_path / "kp.jsonl", 8, 5, log=lambda m: None)
    demo_cfg.keyphrase.method = "precomputed"
    demo_cfg.keyphrase.query_method = "precomputed"
    demo_cfg.keyphrase.query_fallback = "seq2seq"
    demo_cfg.keyphrase.precomputed_files = [str(tmp_path / "kp.jsonl")]
    r = HybridRetriever.build(demo_cfg, demo_dataset.documents)
    assert all(kps for kps in r.doc_keyphrases)
    known = demo_dataset.queries["q1"]
    free = "completely new free text query about graphs"
    res_known, res_free = r.search(known, top_k=3), r.search(free, top_k=3)
    assert res_known.query_keyphrases[-1][0] == "shared topic"       # berasal dari berkas precomputed
    assert res_free.query_keyphrases == [("completely new", 1.0)]          # dari generator cadangan (tiruan)
    r.save()
    assert HybridRetriever.load(demo_cfg).search(known, top_k=3).hits[0].doc_id == res_known.hits[0].doc_id


def test_precomputed_docs_are_strict(demo_cfg, demo_dataset, tmp_path):
    from hybrid_ir.keyphrases import MissingPrecomputed
    from hybrid_ir.retriever import HybridRetriever
    (tmp_path / "empty.jsonl").write_text("")
    demo_cfg.keyphrase.method = "precomputed"
    demo_cfg.keyphrase.precomputed_files = [str(tmp_path / "empty.jsonl")]
    with pytest.raises(MissingPrecomputed):
        HybridRetriever.build(demo_cfg, demo_dataset.documents)


def test_resolve_data_dir(gen, train, tmp_path):
    for mod in (gen, train):
        base = tmp_path / mod.__name__
        direct = base / "direct"
        direct.mkdir(parents=True)
        (direct / "corpus.jsonl").write_text("{}")
        assert mod.resolve_data_dir(direct) == direct
        nested = base / "input" / "datasets" / "alice" / "hybrid-ir-scifact"
        nested.mkdir(parents=True)
        (nested / "corpus.jsonl").write_text("{}")
        assert mod.resolve_data_dir("/kaggle/input/hybrid-ir-scifact", search_root=base / "input") == nested
        with pytest.raises(SystemExit):
            mod.resolve_data_dir("/nope", search_root=base / "kosong")
