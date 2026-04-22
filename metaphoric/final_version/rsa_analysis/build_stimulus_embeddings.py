"""
build_stimulus_embeddings.py

用途
- 为 Model-RSA 中的 M3（预训练语义 RDM）准备一次性预计算的词向量表。
- 从刺激模板中抽取所有唯一 `word_label`，用 bert-base-chinese mean-pooling
  得到 768 维向量；无网络环境下可用 `--embedding-source random` 跑通流程。

输入
- `--stimuli-template`：刺激模板 CSV（默认读取 `rsa_config.STIMULI_TEMPLATE`）
  需含 `word_label` 列（若列名不同可通过 `--word-col` 覆盖）。

输出
- `stimulus_embeddings_bert.tsv`：列 `word_label`, `dim_0`, `dim_1`, ..., `dim_N`
- `manifest.json`：记录 source/model_name/dim/n_words/timestamp

下游
- `model_rdm_comparison.py --embedding-file stimulus_embeddings_bert.tsv` 用于 M3。

常见坑
- bert-base-chinese 第一次会从 HuggingFace 下载约 400MB 权重；如在离线环境，
  请先在联网机器上下载好缓存再挂载；或使用 `--embedding-source random` 兜底。
- 若 `word_label` 是多字词，BERT 会拆成多个 token，这里使用 mean-pooling。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402


def load_word_list(stimuli_path: Path, word_col: str) -> list[str]:
    suffix = stimuli_path.suffix.lower()
    frame = pd.read_csv(stimuli_path, sep="\t") if suffix in {".tsv", ".txt"} else pd.read_csv(stimuli_path)
    if word_col not in frame.columns:
        raise ValueError(f"Column '{word_col}' not found in {stimuli_path}. Available: {list(frame.columns)}")
    words = frame[word_col].dropna().astype(str).str.strip()
    words = words[words.ne("")]
    unique_words = sorted(set(words.tolist()))
    if not unique_words:
        raise ValueError("No valid words extracted from stimuli template.")
    return unique_words


def embed_with_bert(words: list[str], model_name: str) -> tuple[np.ndarray, str]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers/torch unavailable. Install them or use --embedding-source random."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    vectors = []
    with torch.no_grad():
        for word in words:
            encoded = tokenizer(word, return_tensors="pt", add_special_tokens=True)
            output = model(**encoded)
            hidden = output.last_hidden_state.squeeze(0).numpy()
            mask = encoded["attention_mask"].squeeze(0).numpy().astype(bool)
            vectors.append(hidden[mask].mean(axis=0))
    return np.vstack(vectors), model_name


def embed_with_random(words: list[str], dim: int, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((len(words), dim)).astype(np.float32)
    return matrix, f"random(seed={seed},dim={dim})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stimulus word embeddings for Model-RSA M3.")
    parser.add_argument("--stimuli-template", type=Path, default=None,
                        help="Path to stimuli template (csv/tsv). Defaults to rsa_config.STIMULI_TEMPLATE.")
    parser.add_argument("--word-col", default="word_label")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write embeddings. Defaults to {BASE_DIR}/stimulus_embeddings.")
    parser.add_argument("--output-name", default="stimulus_embeddings_bert.tsv")
    parser.add_argument("--embedding-source", choices=["bert", "random"], default="bert")
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--random-dim", type=int, default=768)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    from rsa_analysis.rsa_config import BASE_DIR, STIMULI_TEMPLATE  # noqa: E402

    stimuli_path = args.stimuli_template or STIMULI_TEMPLATE
    output_dir = ensure_dir(args.output_dir or (BASE_DIR / "stimulus_embeddings"))

    words = load_word_list(stimuli_path, args.word_col)

    if args.embedding_source == "bert":
        matrix, source_tag = embed_with_bert(words, args.model_name)
    else:
        matrix, source_tag = embed_with_random(words, args.random_dim, args.random_seed)

    dim_cols = [f"dim_{i}" for i in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix, columns=dim_cols)
    frame.insert(0, "word_label", words)
    write_table(frame, output_dir / args.output_name)

    manifest = {
        "source": args.embedding_source,
        "model_or_tag": source_tag,
        "dim": int(matrix.shape[1]),
        "n_words": int(matrix.shape[0]),
        "stimuli_template": str(stimuli_path),
        "word_col": args.word_col,
        "output_file": str(output_dir / args.output_name),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(manifest, output_dir / "manifest.json")
    print(f"[build_stimulus_embeddings] wrote {matrix.shape[0]} words x {matrix.shape[1]} dims -> "
          f"{output_dir / args.output_name}")


if __name__ == "__main__":
    main()
