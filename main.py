
# !pip install -q transformers==4.44.2 accelerate==0.34.2 sentencepiece

import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import TruncatedSVD

# ----------------- USER SETTINGS -----------------
model_name = "EleutherAI/gpt-neo-1.3B"
# model_name = "facebook/opt-1.3b"
# model_name = "google/gemma-2-2b"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#pick your model based on size constraints

device = "cuda" if torch.cuda.is_available() else "cpu"
k = 30
input_csv = Path("qasamples.csv")   # can be a directory or a file
output_csv = Path("gptk30.csv")
max_length = 512


EPS = 1e-12



def compute_laplacian_diag_from_attention(A: np.ndarray):
    col_sum = A.sum(axis=0)
    col_nonzero_counts = (np.abs(A) > EPS).sum(axis=0).astype(float)
    col_nonzero_counts[col_nonzero_counts == 0.0] = 1.0
    d_diag = col_sum / col_nonzero_counts
    A_diag = np.diag(A) if A.shape[0] == A.shape[1] else np.zeros(A.shape[0])
    lap_diag = d_diag - A_diag
    return lap_diag

def top_k_from_diag(diag_vec: np.ndarray, k: int):
    T = diag_vec.shape[0]
    if T <= 0:
        return np.zeros(k, dtype=float)
    sorted_vals = np.sort(diag_vec)
    if k <= T:
        topk = sorted_vals[-k:][::-1]
    else:
        topk_present = sorted_vals[::-1]
        pad = np.zeros(k - T, dtype=float)
        topk = np.concatenate([topk_present, pad])
    return topk

def attention_entropy(A: np.ndarray):
    A_clipped = np.clip(A, EPS, 1.0)
    row_entropy = -np.sum(A_clipped * np.log(A_clipped), axis=-1)
    return float(np.mean(row_entropy)), float(np.var(row_entropy))

def cross_layer_similarity(prev_layer: np.ndarray, curr_layer: np.ndarray):
    if prev_layer is None:
        return 0.0, 0.0
    p = prev_layer.reshape(prev_layer.shape[0], -1)
    c = curr_layer.reshape(curr_layer.shape[0], -1)
    p_norm = np.linalg.norm(p, axis=1, keepdims=True) + EPS
    c_norm = np.linalg.norm(c, axis=1, keepdims=True) + EPS
    p_unit = p / p_norm
    c_unit = c / c_norm
    sim = p_unit @ c_unit.T
    mean_sim = float(np.mean(sim))
    min_sim = float(np.min(sim))
    return mean_sim, min_sim

def safe_svd_reduce_matrix(X: np.ndarray, target_dim: int = 512):
    if X is None:
        return np.zeros(target_dim, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.zeros(target_dim, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_rows, n_cols = X.shape
    n_components = min(target_dim, n_rows, n_cols)
    if n_components <= 0:
        return np.zeros(target_dim, dtype=float)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    try:
        transformed = svd.fit_transform(X)
    except Exception:
        flat = X.flatten()
        if flat.size >= target_dim:
            return flat[:target_dim]
        else:
            return np.pad(flat, (0, target_dim - flat.size))
    flat_transformed = transformed.flatten()
    if flat_transformed.size >= target_dim:
        return flat_transformed[:target_dim]
    else:
        return np.pad(flat_transformed, (0, target_dim - flat_transformed.size))



def extract_per_sample(model, tokenizer, q: str, a: str, k: int, max_length: int = 512, print_info=False):
    text = f"Q: {q}\nA: {a}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)

    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        return np.zeros(512), np.zeros(512), np.zeros(1024)

    L = len(attentions)
    H = attentions[0].shape[1]

    lap_list = []
    e_list = []
    prev_layer_np = None

    for layer_attn in attentions:
        layer_np = layer_attn[0].detach().cpu().numpy()
        for head_idx in range(layer_np.shape[0]):
            A = layer_np[head_idx]
            row_sums = A.sum(axis=1, keepdims=True) + EPS
            A_norm = A / row_sums
            lap_diag = compute_laplacian_diag_from_attention(A_norm)
            topk = top_k_from_diag(lap_diag, k)
            lap_list.append(topk.astype(float))
            mean_ent, var_ent = attention_entropy(A_norm)
            mean_sim, min_sim = cross_layer_similarity(prev_layer_np, layer_np)
            ratio = float(mean_ent / (mean_sim + EPS))
            e_list.append(np.array([mean_ent, var_ent, mean_sim, min_sim, ratio], dtype=float))
        prev_layer_np = layer_np.copy()

    lap_mat = np.stack(lap_list, axis=0) if len(lap_list) > 0 else np.zeros((0, k))
    e_mat = np.stack(e_list, axis=0) if len(e_list) > 0 else np.zeros((0, 5))

    # Print summary only when requested (first sample)
    if print_info:
        print("\n--- Model Attention Summary (pre-SVD) ---")
        print(f"Number of Layers: {L}")
        print(f"Number of Heads per Layer: {H}")
        print(f"LapEigvals matrix shape before SVD: {lap_mat.shape}")
        print(f"E_all matrix shape before SVD: {e_mat.shape}")

    lap_512 = safe_svd_reduce_matrix(lap_mat, target_dim=512)
    e_512 = safe_svd_reduce_matrix(e_mat, target_dim=512)
    final = np.concatenate([lap_512, e_512], axis=0)

    if print_info:
        print("\n--- Model Attention Summary (post-SVD) ---")
        print(f"LapEigvals vector shape after SVD: {lap_512.shape}")
        print(f"E_all vector shape after SVD: {e_512.shape}")
        print(f"Final concatenated vector shape: {final.shape}")
        print("--------------------------------")

    return lap_512, e_512, final




def main():
    print(f"Device: {device}")
    print(f"Loading model: {model_name} ...")


    if not input_csv.exists():
        raise FileNotFoundError(f"Input path not found: {input_csv.resolve()}")
    if input_csv.is_dir():
        # find first .csv file in directory
        csv_files = list(input_csv.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_csv.resolve()}")
        csv_path = csv_files[0]
        print(f"Using CSV file: {csv_path.name}")
    else:
        csv_path = input_csv

    # Safe config + tokenizer loading with fallback to avoid the 'additional_chat_templates' 404
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.output_attentions = True

    # Try robust tokenizer load (avoid chat-template fetch issues)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,          # often avoids the additional_chat_templates call
            trust_remote_code=True,
            revision="main"
        )
    except Exception as tok_err:
        # fallback: try without trust_remote_code and without revision
        print("Tokenizer load fallback triggered:", tok_err)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Load model (keeps your config.output_attentions=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)
    except Exception as e:
        print("Model load failed with exception:", e)
        # Last-resort: try without passing config
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()


    df = pd.read_csv(csv_path)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Input CSV must have 'question' and 'answer' columns.")

    results, lap_cols, e_cols = [], [], []

    print(f"Processing {len(df)} rows from: {csv_path.name}")

    for i, row in enumerate(tqdm(list(df.itertuples(index=False)), total=len(df), desc="Rows")):
        q = str(row.question)
        a = str(row.answer)
        try:
            lap512, e512, final1024 = extract_per_sample(
                model, tokenizer, q, a, k, max_length=max_length, print_info=(i == 0)
            )
        except Exception as exc:
            print(f"Error processing row {i}: {exc}")
            lap512 = np.zeros(512, dtype=float)
            e512 = np.zeros(512, dtype=float)
            final1024 = np.zeros(1024, dtype=float)
        results.append(final1024.tolist())
        lap_cols.append(lap512.tolist())
        e_cols.append(e512.tolist())

    df_out = df.copy()
    df_out["lap_512"] = lap_cols
    df_out["e_512"] = e_cols
    df_out["final_vector"] = results

    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved output to: {output_csv.resolve()}")
    print("Each 'final_vector' is length:", len(df_out.iloc[0]["final_vector"]))

if __name__ == "__main__":
    main()