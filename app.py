# app.py — Hybrid boosted matcher (local CSV only)
import os
import io
import tempfile
import re
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# -------- CONFIG --------
CSV_PATH = "dataset_updated.csv"
IMAGE_FOLDER = "sample"
TOP_K = 5
DEFAULT_W_TEXT = 0.7
COLOR_BINS = (16, 16, 8)
SPATIAL_GRID = (2, 2)
HIST_TARGET_SIZE = (256, 256)
COLOR_POWER_BOOST = 1.7

st.set_page_config(layout="wide", page_title="Lost & Found — Local Hybrid Matcher")

# -------- Utilities --------
def normalize_text(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\'s\b", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8').fillna("").astype(str)
    except Exception:
        try:
            return pd.read_csv(path, encoding='latin-1').fillna("").astype(str)
        except Exception:
            raise

def load_dataset(local_path=CSV_PATH):
    if os.path.exists(local_path):
        try:
            df = safe_read_csv(local_path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    expected = ['entry_id','type','category','color','description','image_filename']
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    for c in expected:
        df[c] = df[c].fillna("").astype(str)
    return df

# -------- Image helpers --------
def safe_imread_pil(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def pil_to_cv2(pil):
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess_image_for_hist(image_cv2, target_size=HIST_TARGET_SIZE):
    if image_cv2 is None:
        return None
    try:
        img = cv2.resize(image_cv2, target_size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(v)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception:
        return image_cv2

def compute_global_hist(image_cv2, bins=COLOR_BINS):
    if image_cv2 is None:
        return np.zeros(bins[0]*bins[1]*bins[2], dtype=float)
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2], None, bins, [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compute_spatial_hist(image_cv2, bins=COLOR_BINS, grid=SPATIAL_GRID):
    if image_cv2 is None:
        return np.zeros((bins[0]*bins[1]*bins[2]*grid[0]*grid[1]), dtype=float)
    h, w = image_cv2.shape[:2]
    chists = []
    for gy in range(grid[0]):
        for gx in range(grid[1]):
            y0 = int(h * gy / grid[0]); y1 = int(h * (gy+1) / grid[0])
            x0 = int(w * gx / grid[1]); x1 = int(w * (gx+1) / grid[1])
            cell = image_cv2[y0:y1, x0:x1]
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv],[0,1,2], None, bins, [0,180,0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            chists.append(hist)
    if len(chists) == 0:
        return np.zeros((bins[0]*bins[1]*bins[2]*grid[0]*grid[1]), dtype=float)
    concat = np.concatenate(chists)
    norm = np.linalg.norm(concat)
    if norm > 0:
        concat = concat / norm
    return concat

def compute_color_moments(image_cv2):
    if image_cv2 is None:
        return np.zeros(9, dtype=float)
    img = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB).astype(np.float32)
    feats = []
    for ch in range(3):
        c = img[:,:,ch].ravel()
        mean = float(np.mean(c)); std = float(np.std(c))
        skew = 0.0 if std < 1e-8 else float(np.mean(((c-mean)/std)**3))
        feats.extend([mean, std, skew])
    return np.array(feats, dtype=float)

def compute_color_features(image_cv2, bins=COLOR_BINS, grid=SPATIAL_GRID):
    if image_cv2 is None:
        return (np.zeros(np.prod(bins)), np.zeros(np.prod(bins)*grid[0]*grid[1]), np.zeros(9))
    img = preprocess_image_for_hist(image_cv2)
    gh = compute_global_hist(img, bins=bins)
    sh = compute_spatial_hist(img, bins=bins, grid=grid)
    cm = compute_color_moments(img)
    return gh, sh, cm

def image_bytes_to_cv2(file_bytes):
    try:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        arr = np.array(pil)
        return pil, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None, None

def candidate_image_paths(category, filename):
    base, ext = os.path.splitext(filename)
    candidates = []
    if filename == "":
        return []
    if os.path.isabs(filename) or (os.path.sep in filename):
        candidates.append(filename)
    if category:
        candidates.append(os.path.join(IMAGE_FOLDER, category, filename))
        for e in ['.jpg','.jpeg','.png']:
            candidates.append(os.path.join(IMAGE_FOLDER, category, base + e))
    candidates.append(os.path.join(IMAGE_FOLDER, filename))
    for e in ['.jpg','.jpeg','.png']:
        candidates.append(os.path.join(IMAGE_FOLDER, base + e))
    seen = set(); out = []
    for p in candidates:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

# -------- Feature build (cached) ----------
@st.cache_data(ttl=600)
def build_text_and_color_features(df, bins=COLOR_BINS, grid=SPATIAL_GRID):
    if df is None or df.empty:
        tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=700, stop_words='english', sublinear_tf=True)
        return tfidf, np.zeros((0,1)), np.zeros((0, np.prod(bins))), np.zeros((0, np.prod(bins)*grid[0]*grid[1])), np.zeros((0,9))
    descs = df['description'].fillna("").astype(str).tolist()
    preproc = [normalize_text(d) for d in descs]
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=700, stop_words='english', sublinear_tf=True, min_df=1)
    try:
        tfidf_mat = tfidf.fit_transform(preproc).toarray()
        tfidf_norm = normalize(tfidf_mat, norm='l2')
    except Exception:
        tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=500)
        tfidf_mat = tfidf.fit_transform(preproc).toarray()
        tfidf_norm = normalize(tfidf_mat, norm='l2')

    color_globals = []; color_spatials = []; color_moms = []; missing = []
    for idx, row in df.iterrows():
        fn = str(row.get('image_filename','')).strip()
        cat = str(row.get('category','')).strip()
        if fn == "":
            missing.append((idx, fn, ["<no filename>"]))
            color_globals.append(np.zeros(np.prod(bins), dtype=float))
            color_spatials.append(np.zeros(np.prod(bins)*grid[0]*grid[1], dtype=float))
            color_moms.append(np.zeros(9, dtype=float))
            continue
        paths = candidate_image_paths(cat, fn)
        found = None
        for p in paths:
            if os.path.exists(p):
                found = p; break
        if found is None:
            missing.append((idx, fn, paths))
            color_globals.append(np.zeros(np.prod(bins), dtype=float))
            color_spatials.append(np.zeros(np.prod(bins)*grid[0]*grid[1], dtype=float))
            color_moms.append(np.zeros(9, dtype=float))
        else:
            pil = safe_imread_pil(found)
            cv2img = pil_to_cv2(pil) if pil is not None else None
            gh, sh, cm = compute_color_features(cv2img, bins=bins, grid=grid)
            color_globals.append(gh); color_spatials.append(sh); color_moms.append(cm)

    color_globals = np.vstack(color_globals) if len(color_globals)>0 else np.zeros((0,np.prod(bins)))
    color_spatials = np.vstack(color_spatials) if len(color_spatials)>0 else np.zeros((0,np.prod(bins)*grid[0]*grid[1]))
    color_moms = np.vstack(color_moms) if len(color_moms)>0 else np.zeros((0,9))

    color_globals_n = normalize(color_globals, norm='l2') if color_globals.size else color_globals
    color_spatials_n = normalize(color_spatials, norm='l2') if color_spatials.size else color_spatials

    return tfidf, tfidf_norm, color_globals_n, color_spatials_n, color_moms

# -------- Similarity helpers ----------
def chi2_sim(a,b,eps=1e-10):
    if a is None or b is None or a.size==0 or b.size==0:
        return 0.0
    num = (a-b)**2
    den = a+b+eps
    chi = 0.5 * np.sum(num/den)
    return float(np.exp(-chi))

def hist_intersection(a,b):
    if a is None or b is None or a.size==0 or b.size==0:
        return 0.0
    return float(np.sum(np.minimum(a,b)))

def safe_minmax(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# -------- Ranker (boosted fusion) ----------
def rank_candidates_boosted(q_tfidf, q_color_tuple, tfidf_found, color_global_found, color_spatial_found, color_mom_found, found_df, top_k=5, w_text=DEFAULT_W_TEXT):
    n = len(found_df)
    if n == 0:
        return []

    # text sim
    if tfidf_found.size == 0 or q_tfidf is None or q_tfidf.size==0:
        sim_text = np.zeros(n)
    else:
        try:
            sim_text = cosine_similarity(q_tfidf.reshape(1,-1), tfidf_found).reshape(-1)
        except Exception:
            sim_text = np.zeros(n)

    # color sims
    q_global, q_spatial, q_mom = (None, None, None)
    if q_color_tuple is not None:
        q_global, q_spatial, q_mom = q_color_tuple

    sim_g_cos = np.zeros(n); sim_g_inter = np.zeros(n); sim_g_chi = np.zeros(n)
    sim_s_cos = np.zeros(n); sim_cm = np.zeros(n)
    for i in range(n):
        if color_global_found.size:
            a = color_global_found[i]
            if q_global is not None and a is not None and q_global.size and a.size:
                try:
                    sim_g_cos[i] = cosine_similarity(q_global.reshape(1,-1), a.reshape(1,-1)).reshape(-1)[0]
                except Exception:
                    sim_g_cos[i] = 0.0
                sim_g_inter[i] = hist_intersection(q_global, a)
                sim_g_chi[i] = chi2_sim(q_global, a)
        if color_spatial_found.size:
            b = color_spatial_found[i]
            if q_spatial is not None and b is not None and q_spatial.size and b.size:
                try:
                    sim_s_cos[i] = cosine_similarity(q_spatial.reshape(1,-1), b.reshape(1,-1)).reshape(-1)[0]
                except Exception:
                    sim_s_cos[i] = 0.0
        if color_mom_found.size and q_mom is not None:
            sim_cm[i] = 1.0 / (1.0 + np.mean(np.abs(q_mom - color_mom_found[i])))

    # normalize modalities
    g_cos_n = safe_minmax(sim_g_cos)
    g_inter_n = safe_minmax(sim_g_inter)
    g_chi_n = safe_minmax(sim_g_chi)
    s_cos_n = safe_minmax(sim_s_cos)
    cm_n = safe_minmax(sim_cm)

    # combine color
    sim_color = 0.40 * g_cos_n + 0.25 * g_inter_n + 0.20 * s_cos_n + 0.15 * cm_n - 0.05 * g_chi_n
    sim_color = np.clip(sim_color, 0.0, 1.0)

    sim_text_n = safe_minmax(sim_text)

    # adaptive weighting
    w_t = float(w_text)
    has_text = (q_tfidf is not None and q_tfidf.size>0 and np.sum(q_tfidf)>0)
    has_image = (q_color_tuple is not None and q_color_tuple[0] is not None and np.sum(np.abs(q_color_tuple[0]))>0)
    if not has_text and has_image:
        w_t = 0.0
    if not has_image and has_text:
        w_t = 1.0
    if (not has_text) and has_image:
        sim_color = np.minimum(sim_color * COLOR_POWER_BOOST, 1.0)

    sim_fused = w_t * sim_text_n + (1.0 - w_t) * sim_color

    idxs = np.argsort(sim_fused)[::-1][:top_k]
    results = []
    for rank, i in enumerate(idxs, start=1):
        row = found_df.iloc[i]
        results.append({
            "rank": rank,
            "found_entry_id": row['entry_id'],
            "filename": row['image_filename'],
            "category": row.get('category', ""),
            "color": row.get('color', ""),
            "score": float(sim_fused[i]),
            "sim_text": float(sim_text_n[i]) if np.isfinite(sim_text_n[i]) else 0.0,
            "sim_color": float(sim_color[i]) if np.isfinite(sim_color[i]) else 0.0
        })
    return results

# -------- App UI & Flow --------
st.title("Lost & Found — Local Hybrid Matcher")
st.markdown("TF-IDF (text) + HSV color (global + spatial + moments) fusion with boosting. Uses only local CSV and local images.")

# Load dataset & features
df = load_dataset(CSV_PATH)
if df.empty:
    st.warning(f"Dataset empty or not found at: {CSV_PATH}. You can still upload found items (they'll be saved locally).")
else:
    st.sidebar.write(f"Loaded dataset rows: {len(df)}")
expected_cols = ['entry_id','type','category','color','description','image_filename']
for c in expected_cols:
    if c not in df.columns:
        df[c] = ""
for c in expected_cols:
    df[c] = df[c].fillna("").astype(str)

# Sidebar settings
st.sidebar.header("Settings / Diagnostics")
top_k = st.sidebar.number_input("Top-K", min_value=1, max_value=20, value=TOP_K)
w_text = st.sidebar.slider("Text weight (w_text)", 0.0, 1.0, float(DEFAULT_W_TEXT), 0.05)
st.sidebar.write(f"Image folder: {IMAGE_FOLDER}")
st.sidebar.write(f"CSV file: {CSV_PATH}")

# Build features
tfidf_vec, tfidf_mat, color_global_mat, color_spatial_mat, color_mom_mat = build_text_and_color_features(df, bins=COLOR_BINS, grid=SPATIAL_GRID)
st.sidebar.write(f"TF-IDF shape: {tfidf_mat.shape}")
st.sidebar.write(f"Color global shape: {color_global_mat.shape}")
st.sidebar.write(f"Color spatial shape: {color_spatial_mat.shape}")

# Prepare found subset
type_clean = df['type'].astype(str).str.lower().str.strip()
found_mask = type_clean == 'found'
found_df = df[found_mask].reset_index(drop=True)
found_idxs_orig = df.index[found_mask].tolist()
if len(found_idxs_orig) == 0:
    tfidf_found = np.zeros((0, tfidf_mat.shape[1])) if tfidf_mat.size else np.zeros((0,0))
    color_global_found = np.zeros((0, color_global_mat.shape[1])) if color_global_mat.size else np.zeros((0,0))
    color_spatial_found = np.zeros((0, color_spatial_mat.shape[1])) if color_spatial_mat.size else np.zeros((0,0))
    color_mom_found = np.zeros((0, color_mom_mat.shape[1])) if color_mom_mat.size else np.zeros((0,0))
else:
    tfidf_found = tfidf_mat[found_idxs_orig] if tfidf_mat.size else np.zeros((len(found_idxs_orig),0))
    color_global_found = color_global_mat[found_idxs_orig] if color_global_mat.size else np.zeros((len(found_idxs_orig),0))
    color_spatial_found = color_spatial_mat[found_idxs_orig] if color_spatial_mat.size else np.zeros((len(found_idxs_orig),0))
    color_mom_found = color_mom_mat[found_idxs_orig] if color_mom_mat.size else np.zeros((len(found_idxs_orig),0))

# -------- Upload new FOUND item (local save only) --------
st.header("Upload New FOUND Item (saved locally)")
with st.expander("Add a new FOUND item"):
    new_file = st.file_uploader("Found image (jpg/png)", type=['jpg','jpeg','png'], key="found_up")
    new_cat = st.text_input("Category (optional)")
    new_col = st.text_input("Color (optional)")
    new_desc = st.text_area("Description (optional)")
    save_btn = st.button("Save Found Item")
    if save_btn:
        if new_file is None:
            st.error("Please upload an image.")
        else:
            try:
                os.makedirs(IMAGE_FOLDER, exist_ok=True)
                # compute next filename as found_N.ext (count existing found in dataset)
                try:
                    found_count = int(type_clean.eq('found').sum()) if (df is not None and not df.empty) else 0
                except Exception:
                    found_count = 0
                ext = os.path.splitext(new_file.name)[1].lower()
                if ext not in ['.jpg','.jpeg','.png']:
                    ext = '.jpg'
                idx = found_count + 1
                fname = f"found_{idx}{ext}"
                save_path = os.path.join(IMAGE_FOLDER, fname)
                while os.path.exists(save_path):
                    idx += 1
                    fname = f"found_{idx}{ext}"
                    save_path = os.path.join(IMAGE_FOLDER, fname)
                with open(save_path, "wb") as f:
                    f.write(new_file.getbuffer())

                new_id = f"found_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                new_row = {
                    "entry_id": new_id,
                    "type": "found",
                    "category": new_cat or "",
                    "color": new_col or "",
                    "description": new_desc or "",
                    "image_filename": fname
                }

                # append to CSV atomically
                if os.path.exists(CSV_PATH):
                    df_existing = safe_read_csv(CSV_PATH)
                    df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    df_existing = pd.DataFrame([new_row], columns=expected_cols)
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=os.path.dirname(os.path.abspath(CSV_PATH)) or ".")
                os.close(tmp_fd)
                df_existing.to_csv(tmp_path, index=False)
                os.replace(tmp_path, CSV_PATH)
                st.success(f"Saved found item as {fname} and appended to {CSV_PATH}")

                # reload dataset and features in-memory
                df = load_dataset(CSV_PATH)
                tfidf_vec, tfidf_mat, color_global_mat, color_spatial_mat, color_mom_mat = build_text_and_color_features(df, bins=COLOR_BINS, grid=SPATIAL_GRID)
                # refresh found subset
                type_clean = df['type'].astype(str).str.lower().str.strip()
                found_mask = type_clean == 'found'
                found_df = df[found_mask].reset_index(drop=True)
                found_idxs_orig = df.index[found_mask].tolist()
                if len(found_idxs_orig) == 0:
                    tfidf_found = np.zeros((0, tfidf_mat.shape[1])) if tfidf_mat.size else np.zeros((0,0))
                    color_global_found = np.zeros((0, color_global_mat.shape[1])) if color_global_mat.size else np.zeros((0,0))
                    color_spatial_found = np.zeros((0, color_spatial_mat.shape[1])) if color_spatial_mat.size else np.zeros((0,0))
                    color_mom_found = np.zeros((0, color_mom_mat.shape[1])) if color_mom_mat.size else np.zeros((0,0))
                else:
                    tfidf_found = tfidf_mat[found_idxs_orig] if tfidf_mat.size else np.zeros((len(found_idxs_orig),0))
                    color_global_found = color_global_mat[found_idxs_orig] if color_global_mat.size else np.zeros((len(found_idxs_orig),0))
                    color_spatial_found = color_spatial_mat[found_idxs_orig] if color_spatial_mat.size else np.zeros((len(found_idxs_orig),0))
                    color_mom_found = color_mom_mat[found_idxs_orig] if color_mom_mat.size else np.zeros((len(found_idxs_orig),0))

            except Exception as e:
                st.error("Save failed: " + str(e))

# -------- Query (lost item) --------
st.header("Query (lost item)")
col1, col2 = st.columns([1,2])
with col1:
    uploaded = st.file_uploader("Upload image (optional)", type=['jpg','jpeg','png'], key="query_up")
    description = st.text_area("Description (optional)", placeholder="e.g. blue water bottle with dent on side")
    find_btn = st.button("Find Matches")
with col2:
    st.info("Tip: combine a short description and an image. Use sidebar Text weight slider to tune fusion.")

if find_btn:
    if (uploaded is None) and (not str(description).strip()):
        st.warning("Please upload an image or enter a description.")
    else:
        # rebuild features (in case dataset changed)
        tfidf_vec, tfidf_mat, color_global_mat, color_spatial_mat, color_mom_mat = build_text_and_color_features(df, bins=COLOR_BINS, grid=SPATIAL_GRID)
        # refresh found subset
        type_clean = df['type'].astype(str).str.lower().str.strip()
        found_mask = type_clean == 'found'
        found_df = df[found_mask].reset_index(drop=True)
        found_idxs_orig = df.index[found_mask].tolist()
        if len(found_idxs_orig) == 0:
            tfidf_found = np.zeros((0, tfidf_mat.shape[1])) if tfidf_mat.size else np.zeros((0,0))
            color_global_found = np.zeros((0, color_global_mat.shape[1])) if color_global_mat.size else np.zeros((0,0))
            color_spatial_found = np.zeros((0, color_spatial_mat.shape[1])) if color_spatial_mat.size else np.zeros((0,0))
            color_mom_found = np.zeros((0, color_mom_mat.shape[1])) if color_mom_mat.size else np.zeros((0,0))
        else:
            tfidf_found = tfidf_mat[found_idxs_orig] if tfidf_mat.size else np.zeros((len(found_idxs_orig),0))
            color_global_found = color_global_mat[found_idxs_orig] if color_global_mat.size else np.zeros((len(found_idxs_orig),0))
            color_spatial_found = color_spatial_mat[found_idxs_orig] if color_spatial_mat.size else np.zeros((len(found_idxs_orig),0))
            color_mom_found = color_mom_mat[found_idxs_orig] if color_mom_mat.size else np.zeros((len(found_idxs_orig),0))

        # text feature
        q_desc = description if description is not None else ""
        try:
            q_t = tfidf_vec.transform([normalize_text(q_desc)]).toarray()
            q_tfidf = normalize(q_t, norm='l2')[0] if q_t.size else np.zeros(0)
        except Exception:
            q_tfidf = np.zeros(tfidf_found.shape[1]) if tfidf_found.size else np.zeros(0)

        # image feature
        if uploaded is not None:
            fb = uploaded.read()
            pil_q, cv2_q = image_bytes_to_cv2(fb)
            if cv2_q is not None:
                q_global, q_spatial, q_mom = compute_color_features(cv2_q, bins=COLOR_BINS, grid=SPATIAL_GRID)
                if np.linalg.norm(q_global) > 0:
                    q_global = q_global / (np.linalg.norm(q_global) + 1e-12)
            else:
                q_global, q_spatial, q_mom = None, None, None
        else:
            pil_q, cv2_q = None, None
            q_global, q_spatial, q_mom = None, None, None

        # get results
        results = rank_candidates_boosted(q_tfidf, (q_global, q_spatial, q_mom), tfidf_found, color_global_found, color_spatial_found, color_mom_found, found_df, top_k=top_k, w_text=w_text)

        # display
        st.subheader("Query")
        q1, q2 = st.columns([1,3])
        with q1:
            if pil_q is not None:
                st.image(pil_q, use_column_width=True)
            else:
                st.write("No image uploaded")
        with q2:
            st.markdown("**Description:**")
            st.write(q_desc or "—")

        st.subheader(f"Top-{top_k} candidate matches")
        if not results:
            st.write("No candidate matches found.")
        for res in results:
            a,b = st.columns([1,3])
            with a:
                fpath = None
                for p in candidate_image_paths(res.get('category',''), res.get('filename','')):
                    if os.path.exists(p):
                        fpath = p; break
                if fpath:
                    st.image(fpath, width=150)
                else:
                    st.write("Image not found")
            with b:
                st.markdown(f"**Rank {res['rank']} — {res['found_entry_id']}**")
                st.write(f"Category: {res['category']} — Color: {res['color']}")
                st.write(f"Score: {res['score']:.4f} (text: {res.get('sim_text',0):.4f}, color: {res.get('sim_color',0):.4f})")
                cols = st.columns([1,1,1])
                if cols[0].button(f"Confirm {res['found_entry_id']}", key=f"confirm_{res['found_entry_id']}"):
                    lost_id = f"query_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                    fb_path = "feedback.csv"
                    rec = pd.DataFrame([{"lost_entry_id": lost_id, "found_entry_id": res['found_entry_id'], "confirmed_by": "streamlit_ui"}])
                    if os.path.exists(fb_path):
                        rec.to_csv(fb_path, mode='a', index=False, header=False)
                    else:
                        rec.to_csv(fb_path, index=False)
                    st.success("Saved to feedback.csv")
                if cols[1].button(f"Reject {res['found_entry_id']}", key=f"reject_{res['found_entry_id']}"):
                    st.info("Rejected (no action saved).")
                if cols[2].button(f"View row {res['found_entry_id']}", key=f"view_{res['found_entry_id']}"):
                    st.json(found_df[found_df['entry_id']==res['found_entry_id']].to_dict(orient='records'))

# recent feedback preview
st.sidebar.header("Recent feedback")
if os.path.exists("feedback.csv"):
    try:
        fb = pd.read_csv("feedback.csv")
        st.sidebar.table(fb.tail(8))
    except Exception:
        st.sidebar.write("feedback.csv exists but cannot be shown.")
else:
    st.sidebar.write("No feedback yet.")
