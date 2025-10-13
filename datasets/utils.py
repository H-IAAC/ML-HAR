import errno
import hashlib
import os
import os.path
import csv
import copy
from torch.utils.model_zoo import tqdm
import torch
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, Any  # use Tuple if you want to annotate the return
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  

def to_numpy_3d(x, channels_last: Optional[bool] = None):
    """
    Converte x (PyTorch/TF/NumPy) para np.ndarray (N, C, T).
    Se channels_last=True, espera (N, T, C) e transpõe para (N, C, T).
    Se channels_last=False, assume (N, C, T).
    Se channels_last=None, não transpõe.
    """
    # PyTorch
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().cpu().contiguous().numpy()
    except Exception:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            x = x.numpy()
    except Exception:
        pass

    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Esperado tensor/array 3D, recebi shape={x.shape}")

    if channels_last is True:
        x = np.transpose(x, (0, 2, 1))  # (N, T, C) -> (N, C, T)
    elif channels_last is False:
        pass  # já está (N, C, T)

    return x.astype(np.float32, copy=False)

def assert_consistency(report, tol_train=1e-4, tol_test=1e-4, require_test=True):
    c = report["consistency_with_mu_sigma"]
    if c["max_abs_diff_train"] is None or c["max_abs_diff_train"] > tol_train:
        raise AssertionError(f"Train consistency error: {c['max_abs_diff_train']}")
    if require_test and (c["max_abs_diff_test"] is None or c["max_abs_diff_test"] > tol_test):
        raise AssertionError(f"Test consistency error: {c['max_abs_diff_test']}")


def standardization_pass_fail(
    report,
    *,
    tol_consistency=1e-4,
    # fraction policy (calibrate on unseen-person validation!)
    min_fraction=0.80,       # require ≥80% channels OK
    tol_mean_ch=0.5,         # per-channel |mean| tolerance
    tol_std_ch=0.5,          # per-channel |std-1| tolerance
    # global sanity caps (very loose)
    max_abs_mean_cap=1.5,    # cap on test mean_abs_max
    max_std_dev_cap=1.0      # cap on |std_mean - 1|
):
    T = report["sanity"]["train"]
    test = report["sanity"].get("test")
    C = len(report["fit_stats"]["mu_train"])

    # 1) train must be ~0/1
    ok_train = not T["out_of_tolerance_channels"]

    # 2) numeric consistency with (X-μ)/σ
    Cc = report["consistency_with_mu_sigma"]
    ok_cons = (
        (Cc["max_abs_diff_train"] is None or Cc["max_abs_diff_train"] <= tol_consistency) and
        (Cc["max_abs_diff_test"]  is None or Cc["max_abs_diff_test"]  <= tol_consistency)
    )

    # 3) shift-tolerant test gate
    if test is None:
        ok_test, note = True, "(no test)"
    else:
        # Prefer recomputing with your tolerances if per-channel stats are present
        mu_te = test.get("_mu_per_channel")
        sd_te = test.get("_sd_per_channel")
        if (mu_te is not None) and (sd_te is not None):
            import numpy as np
            bad_mask = (np.abs(mu_te) > tol_mean_ch) | (np.abs(sd_te - 1.0) > tol_std_ch)
            bad_count = int(np.sum(bad_mask))
            frac_ok = 1.0 - bad_count / C
        else:
            bad = test["out_of_tolerance_channels"]  # assumes checker used same tolerances
            bad_count = len(bad)
            frac_ok = 1.0 - bad_count / C

        ok_frac = frac_ok >= min_fraction

        # global caps (coarse)
        ok_caps = (test["mean_abs_max"] <= max_abs_mean_cap) and (abs(test["std_mean"] - 1.0) <= max_std_dev_cap)

        ok_test = ok_frac and ok_caps
        note = f"(frac_ok={frac_ok:.2f}, bad={bad_count}/{C}, caps={'OK' if ok_caps else 'FAIL'})"

    overall = ok_train and ok_cons and ok_test
    msg = (
        ("✅ Standardization OK. " if overall else "❌ Failed. ")
        + ("✅ Train OK. " if ok_train else "❌ Train out of tolerance. ")
        + ("✅ Consistent with (X-μ)/σ. " if ok_cons else "❌ Inconsistent with (X-μ)/σ. ")
        + ("" if test is None else (("✅ Test OK " if ok_test else "❌ Test out of tolerance ") + note))
    )
    return overall, msg



def standardization_message(report, *, tol_consistency=1e-4):
    ok_train = not report["sanity"]["train"]["out_of_tolerance_channels"]
    test_block = report["sanity"].get("test")
    ok_test = (test_block is None) or (not test_block["out_of_tolerance_channels"])

    c = report["consistency_with_mu_sigma"]
    ok_consistency = (
        (c["max_abs_diff_train"] is None or c["max_abs_diff_train"] <= tol_consistency) and
        (c["max_abs_diff_test"]  is None or c["max_abs_diff_test"]  <= tol_consistency)
    )

    M = {
        "ok_train": "✅ Train standardization OK.",
        "bad_train": "❌ Train standardization failed (channels out of tolerance).",
        "ok_test": "✅ Test standardization OK.",
        "bad_test": "❌ Test standardization failed (channels out of tolerance).",
        "ok_cons": "✅ Outputs match (X - μ) / σ.",
        "bad_cons": "⚠️ Standardized arrays don’t match (X - μ) / σ.",
        "overall_ok": "✅ All checks passed.",
        "overall_bad": "❌ Some checks failed.",
    }

    parts = []
    parts.append(M["ok_train"] if ok_train else M["bad_train"])
    if test_block is not None:
        parts.append(M["ok_test"] if ok_test else M["bad_test"])
    parts.append(M["ok_cons"] if ok_consistency else M["bad_cons"])

    overall = M["overall_ok"] if (ok_train and ok_test and ok_consistency) else M["overall_bad"]
    return overall + " " + " ".join(parts)


# --- sumarização e checagem ---
def _summarize_channel_stats(X: np.ndarray, name: str) -> Dict[str, float]:
    mu = X.mean(axis=(0, 2))          # por canal
    sd = X.std(axis=(0, 2))
    return {
        "name": name,
        "mean_abs_max": float(np.abs(mu).max()),
        "mean_abs_med": float(np.median(np.abs(mu))),
        "std_min": float(sd.min()),
        "std_mean": float(sd.mean()),
        "std_max": float(sd.max()),
    }



def _apply_time_cut(X: np.ndarray, time_cut_ratio: Optional[float]) -> np.ndarray:
    if not time_cut_ratio or time_cut_ratio <= 0:
        return X
    N, C, T = X.shape
    cut = int(round(T * time_cut_ratio))
    if 2 * cut >= T:
        return X
    return X[:, :, cut:T-cut]

def _per_channel_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # médias e desvios por canal agregando (janela, tempo)
    mu = X.mean(axis=(0, 2))
    sd = X.std(axis=(0, 2))
    return mu, sd

def check_standardization_adapter_all_modes(
    X_train_raw, X_train_z,
    X_test_raw=None, X_test_z=None,
    mu_train=None, sd_train=None,
    *,
    mode="channel",                 # "channel" | "timestep" | "timestep_pooled" | "window_samplewise"
    time_cut_ratio=None,            # used only for "channel" (to mirror your fitter)
    eps=1e-8,
    tol_mean=0.5,
    tol_std=0.5,
    include_maps=False,
):
    assert X_train_raw.ndim == 3 and X_train_z.ndim == 3
    N, C, T = X_train_raw.shape

    def tail_cut_time(X, r):
        if (mode != "channel") or (not r) or (r <= 0):
            return X
        cut = max(1, int(round(T * r)))
        return X[:, :, -cut:]

    Xfit = tail_cut_time(X_train_raw, time_cut_ratio)

    # ---- compute train μ/σ in the SAME way the transform does
    if mu_train is None or sd_train is None:
        if mode == "channel":
            mu_train = Xfit.mean(axis=(0,2), keepdims=True)         # (1,C,1)
            sd_train = Xfit.std(axis=(0,2), keepdims=True) + eps    # (1,C,1)
        elif mode == "timestep":
            mu_train = X_train_raw.mean(axis=0, keepdims=True)      # (1,C,T)
            sd_train = X_train_raw.std(axis=0, keepdims=True) + eps
        elif mode == "timestep_pooled":
            mu_train = X_train_raw.mean(axis=(0,1), keepdims=True)  # (1,1,T)
            sd_train = X_train_raw.std(axis=(0,1), keepdims=True) + eps
        elif mode == "window_samplewise":
            mu_train = X_train_raw.mean(axis=2, keepdims=True)      # (N,C,1)
            sd_train = X_train_raw.std(axis=2, keepdims=True) + eps
        else:
            raise ValueError("unknown mode")
    else:
        mu_train = np.asarray(mu_train)
        sd_train = np.asarray(sd_train)
        
        # force expected shapes for broadcasting
        if mode == "channel":             # per-channel over N×T
            mu_train = mu_train.reshape(1, C, 1)
            sd_train = sd_train.reshape(1, C, 1)
        elif mode == "timestep":          # per (channel,timestep) over N
            mu_train = mu_train.reshape(1, C, T)
            sd_train = sd_train.reshape(1, C, T)
        elif mode == "timestep_pooled":   # per timestep pooled over N×C
            mu_train = mu_train.reshape(1, 1, T)
            sd_train = sd_train.reshape(1, 1, T)
        elif mode == "window_samplewise": # per (sample,channel) over T
            mu_train = mu_train.reshape(N, C, 1)
            sd_train = sd_train.reshape(N, C, 1)
        else:
            raise ValueError("unknown mode")

    # ---- numeric consistency: z = (X - μ)/σ must match X_z
    if mode == "window_samplewise":
        zexp_tr = (X_train_raw - mu_train) / sd_train                      # (N,C,T)
    else:
        zexp_tr = (X_train_raw - mu_train) / sd_train                      # broadcasts
    max_abs_diff_train = float(np.max(np.abs(zexp_tr - X_train_z)))

    max_abs_diff_test = None
    if X_test_raw is not None and X_test_z is not None:
        if mode == "window_samplewise":
            mu_te = X_test_raw.mean(axis=2, keepdims=True)
            sd_te = X_test_raw.std(axis=2, keepdims=True) + eps
            zexp_te = (X_test_raw - mu_te) / sd_te
        else:
            zexp_te = (X_test_raw - mu_train) / sd_train
        max_abs_diff_test = float(np.max(np.abs(zexp_te - X_test_z)))

    # ---- sanity after-standardization (means ~0, std ~1 over the right axis)
    if mode == "channel":
        # check per-channel over N×T
        mu_chk_tr = X_train_z.mean(axis=(0,2))                           # (C,)
        sd_chk_tr = X_train_z.std(axis=(0,2))                            # (C,)
        bad_idx_tr = np.where((np.abs(mu_chk_tr) > tol_mean) | (np.abs(sd_chk_tr - 1) > tol_std))[0]
        names_tr = [f"s{j+1:02d}" for j in range(C)]
        train_block = {
            "unit_count": C,
            "out_of_tolerance": [names_tr[j] for j in bad_idx_tr],
            "mean_abs_max": float(np.abs(mu_chk_tr).max()),
            "std_mean": float(sd_chk_tr.mean()),
        }
        if include_maps:
            train_block["_mu_per_channel"] = mu_chk_tr
            train_block["_sd_per_channel"] = sd_chk_tr

        test_block = None
        if X_test_z is not None:
            mu_chk_te = X_test_z.mean(axis=(0,2))
            sd_chk_te = X_test_z.std(axis=(0,2))
            bad_idx_te = np.where((np.abs(mu_chk_te) > tol_mean) | (np.abs(sd_chk_te - 1) > tol_std))[0]
            names_te = names_tr
            test_block = {
                "unit_count": C,
                "out_of_tolerance": [names_te[j] for j in bad_idx_te],
                "mean_abs_max": float(np.abs(mu_chk_te).max()),
                "std_mean": float(sd_chk_te.mean()),
            }
            if include_maps:
                test_block["_mu_per_channel"] = mu_chk_te
                test_block["_sd_per_channel"] = sd_chk_te

    elif mode == "timestep":
        # check per (channel,timestep) over N
        mu_map_tr = X_train_z.mean(axis=0)                                # (C,T)
        sd_map_tr = X_train_z.std(axis=0)                                 # (C,T)
        bad_mask_tr = (np.abs(mu_map_tr) > tol_mean) | (np.abs(sd_map_tr - 1) > tol_std)
        names = [f"s{c+1:02d}_t{t+1:03d}" for c in range(C) for t in range(T)]
        train_block = {
            "unit_count": C*T,
            "out_of_tolerance": [names[i] for i in np.where(bad_mask_tr.ravel())[0]],
            "mean_abs_max": float(np.abs(mu_map_tr).max()),
            "std_mean": float(sd_map_tr.mean()),
        }
        if include_maps:
            train_block["_mu_ct"] = mu_map_tr
            train_block["_sd_ct"] = sd_map_tr

        test_block = None
        if X_test_z is not None:
            mu_map_te = X_test_z.mean(axis=0)
            sd_map_te = X_test_z.std(axis=0)
            bad_mask_te = (np.abs(mu_map_te) > tol_mean) | (np.abs(sd_map_te - 1) > tol_std)
            test_block = {
                "unit_count": C*T,
                "out_of_tolerance": [names[i] for i in np.where(bad_mask_te.ravel())[0]],
                "mean_abs_max": float(np.abs(mu_map_te).max()),
                "std_mean": float(sd_map_te.mean()),
            }
            if include_maps:
                test_block["_mu_ct"] = mu_map_te
                test_block["_sd_ct"] = sd_map_te

    elif mode == "timestep_pooled":
        # check per timestep pooled over N×C
        mu_t_tr = X_train_z.mean(axis=(0,1))                               # (T,)
        sd_t_tr = X_train_z.std(axis=(0,1))                                # (T,)
        bad_idx_tr = np.where((np.abs(mu_t_tr) > tol_mean) | (np.abs(sd_t_tr - 1) > tol_std))[0]
        names_t = [f"t{t+1:03d}" for t in range(T)]
        train_block = {
            "unit_count": T,
            "out_of_tolerance": [names_t[j] for j in bad_idx_tr],
            "mean_abs_max": float(np.abs(mu_t_tr).max()),
            "std_mean": float(sd_t_tr.mean()),
        }
        if include_maps:
            train_block["_mu_t"] = mu_t_tr
            train_block["_sd_t"] = sd_t_tr

        test_block = None
        if X_test_z is not None:
            mu_t_te = X_test_z.mean(axis=(0,1))
            sd_t_te = X_test_z.std(axis=(0,1))
            bad_idx_te = np.where((np.abs(mu_t_te) > tol_mean) | (np.abs(sd_t_te - 1) > tol_std))[0]
            test_block = {
                "unit_count": T,
                "out_of_tolerance": [names_t[j] for j in bad_idx_te],
                "mean_abs_max": float(np.abs(mu_t_te).max()),
                "std_mean": float(sd_t_te.mean()),
            }
            if include_maps:
                test_block["_mu_t"] = mu_t_te
                test_block["_sd_t"] = sd_t_te

    else:  # window_samplewise
        # check per (sample,channel) over T
        mu_nc_tr = X_train_z.mean(axis=2)                                   # (N,C)
        sd_nc_tr = X_train_z.std(axis=2)                                    # (N,C)
        bad_mask_tr = (np.abs(mu_nc_tr) > tol_mean) | (np.abs(sd_nc_tr - 1) > tol_std)
        names = [f"n{n+1:03d}_s{c+1:02d}" for n in range(N) for c in range(C)]
        train_block = {
            "unit_count": N*C,
            "out_of_tolerance": [names[i] for i in np.where(bad_mask_tr.ravel())[0]],
            "mean_abs_max": float(np.abs(mu_nc_tr).max()),
            "std_mean": float(sd_nc_tr.mean()),
        }
        if include_maps:
            train_block["_mu_nc"] = mu_nc_tr
            train_block["_sd_nc"] = sd_nc_tr

        test_block = None
        if X_test_z is not None:
            mu_nc_te = X_test_z.mean(axis=2)
            sd_nc_te = X_test_z.std(axis=2)
            bad_mask_te = (np.abs(mu_nc_te) > tol_mean) | (np.abs(sd_nc_te - 1) > tol_std)
            test_block = {
                "unit_count": N*C,
                "out_of_tolerance": [names[i] for i in np.where(bad_mask_te.ravel())[0]],
                "mean_abs_max": float(np.abs(mu_nc_te).max()),
                "std_mean": float(sd_nc_te.mean()),
            }
            if include_maps:
                test_block["_mu_nc"] = mu_nc_te
                test_block["_sd_nc"] = sd_nc_te

    return {
        "fit_stats": {"mode": mode, "mu_train": mu_train, "sd_train": sd_train, "time_cut_ratio": time_cut_ratio},
        "consistency_with_mu_sigma": {"max_abs_diff_train": max_abs_diff_train, "max_abs_diff_test": max_abs_diff_test},
        "sanity": {"train": train_block, "test": test_block},
        "tolerances": {"tol_mean": tol_mean, "tol_std": tol_std},
    }


def standardization_pass_fail_all_modes(report, tol_consistency=1e-4, min_fraction=0.80,
                                 max_abs_mean_cap=1.5, max_std_dev_cap=1.0):
    T = report["sanity"]["train"]; Te = report["sanity"].get("test")
    ok_train = (len(T["out_of_tolerance"]) == 0)

    Cc = report["consistency_with_mu_sigma"]
    ok_cons = ((Cc["max_abs_diff_train"] is None or Cc["max_abs_diff_train"] <= tol_consistency) and
               (Cc["max_abs_diff_test"]  is None or Cc["max_abs_diff_test"]  <= tol_consistency))

    if Te is None:
        ok_test, note = True, "(no test)"
    else:
        bad = len(Te["out_of_tolerance"])
        frac_ok = 1.0 - bad / max(1, Te["unit_count"])
        ok_frac = frac_ok >= min_fraction
        ok_caps = (Te["mean_abs_max"] <= max_abs_mean_cap) and (abs(Te["std_mean"] - 1.0) <= max_std_dev_cap)
        ok_test = ok_frac and ok_caps
        note = f"(frac_ok={frac_ok:.2f}, bad={bad}/{Te['unit_count']}, caps={'OK' if ok_caps else 'FAIL'})"

    overall = ok_train and ok_cons and ok_test
    msg = (("✅ OK. " if overall else "❌ Failed. ")
           + ("✅ Train OK. " if ok_train else "❌ Train out of tolerance. ")
           + ("✅ Consistent (X-μ)/σ. " if ok_cons else "❌ Inconsistent (X-μ)/σ. ")
           + ("" if Te is None else (("✅ Test OK " if ok_test else "❌ Test out of tolerance ") + note)))
    return overall, msg





def check_standardization_adapter(
    X_train_raw: np.ndarray,
    X_train_z: np.ndarray,
    X_test_raw: Optional[np.ndarray] = None,
    X_test_z: Optional[np.ndarray] = None,
    mu_train: Optional[np.ndarray] = None,
    sd_train: Optional[np.ndarray] = None,
    *,
    time_cut_ratio: Optional[float] = None,
    eps: float = 1e-8,
    tol_mean_ch: float = 0.5,
    tol_std_ch: float  = 0.5,
    include_per_channel: bool = True,
) -> Dict[str, Any]:
    assert X_train_raw.ndim == 3 and X_train_z.ndim == 3, "Expected (N, C, T)"
    N, C, T = X_train_raw.shape

    # ---- tail cut (matches the fitter)
    def _tail_cut(X: np.ndarray, r: Optional[float]) -> np.ndarray:
        if not r or r <= 0:
            return X
        cut = max(1, int(round(X.shape[2] * r)))
        return X[:, :, -cut:]  # tail only

    def _per_channel_stats(X: np.ndarray):
        return X.mean(axis=(0, 2)), X.std(axis=(0, 2))

    names = [f"s{j+1:02d}" for j in range(C)]  # 1-based labels: s01..sC

    # ---- fit stats (μ, σ) from TRAIN RAW (optionally with tail cut)
    Xfit = _tail_cut(X_train_raw, time_cut_ratio)
    if (mu_train is None) or (sd_train is None):
        mu_train, sd_train = _per_channel_stats(Xfit)              # (C,), (C,)
    else:
        # squeeze any (1,C,1) etc. into (C,)
        mu_train = np.asarray(mu_train).reshape(-1)
        sd_train = np.asarray(sd_train).reshape(-1)
    sd_train = np.where(sd_train < eps, eps, sd_train)

    # ---- numeric consistency with (X - μ)/σ
    zexp_tr = (X_train_raw - mu_train[None, :, None]) / sd_train[None, :, None]
    max_abs_diff_train = float(np.max(np.abs(zexp_tr - X_train_z)))

    max_abs_diff_test = None
    if X_test_raw is not None and X_test_z is not None:
        zexp_te = (X_test_raw - mu_train[None, :, None]) / sd_train[None, :, None]
        max_abs_diff_test = float(np.max(np.abs(zexp_te - X_test_z)))

    # ---- per-channel stats AFTER standardization
    mu_trz, sd_trz = _per_channel_stats(X_train_z)
    bad_train_idx = np.where((np.abs(mu_trz) > tol_mean_ch) | (np.abs(sd_trz - 1.0) > tol_std_ch))[0]

    mu_tez = sd_tez = None
    bad_test_idx = np.array([], dtype=int)
    if X_test_z is not None:
        mu_tez, sd_tez = _per_channel_stats(X_test_z)
        bad_test_idx = np.where((np.abs(mu_tez) > tol_mean_ch) | (np.abs(sd_tez - 1.0) > tol_std_ch))[0]

    train_block = {
        "mean_abs_max": float(np.abs(mu_trz).max()),
        "mean_abs_med": float(np.median(np.abs(mu_trz))),
        "std_min": float(sd_trz.min()),
        "std_mean": float(sd_trz.mean()),
        "std_max": float(sd_trz.max()),
        "out_of_tolerance_channels": [names[j] for j in bad_train_idx],
    }
    if include_per_channel:
        train_block["_mu_per_channel"] = mu_trz
        train_block["_sd_per_channel"] = sd_trz

    test_block = None
    if X_test_z is not None:
        test_block = {
            "mean_abs_max": float(np.abs(mu_tez).max()),
            "mean_abs_med": float(np.median(np.abs(mu_tez))),
            "std_min": float(sd_tez.min()),
            "std_mean": float(sd_tez.mean()),
            "std_max": float(sd_tez.max()),
            "out_of_tolerance_channels": [names[j] for j in bad_test_idx],
        }
        if include_per_channel:
            test_block["_mu_per_channel"] = mu_tez
            test_block["_sd_per_channel"] = sd_tez

    _, sd_raw_fit = _per_channel_stats(Xfit)
    near_const = [names[j] for j in np.where(sd_raw_fit < 1e-6)[0]]

    return {
        "fit_stats": {"mu_train": mu_train, "sd_train": sd_train, "time_cut_ratio": time_cut_ratio},
        "sanity": {"train": train_block, "test": test_block},
        "consistency_with_mu_sigma": {"max_abs_diff_train": max_abs_diff_train, "max_abs_diff_test": max_abs_diff_test},
        "near_constant_channels_raw_train": near_const,
        "tolerances": {"tol_mean_ch": tol_mean_ch, "tol_std_ch": tol_std_ch},
    }


def stats_X(X): 
        return X.mean(dim=(0,2)), X.std(dim=(0,2))


def plot_distribution_over_time(X, sensors=None, names=None, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    n_windows, n_sensors, n_time = X.shape
    if sensors is None:
        sensors = range(n_sensors)
    if names is None:
        names = [f"s{j:02d}" for j in sensors]

    t = np.arange(n_time)

    for j, name in zip(sensors, names):
        q = np.quantile(X[:, j, :], qs, axis=0)  # agrega nas 2324 janelas
        plt.figure(figsize=(12,3.6))
        plt.plot(t, q[2], label="mediana", linewidth=1.2)
        plt.fill_between(t, q[1], q[3], alpha=0.3, label="IQR (25–75%)")
        plt.fill_between(t, q[0], q[4], alpha=0.15, label="5–95%")
        plt.xlabel("tempo (amostras)")
        plt.ylabel("amplitude")
        plt.title(f"Distribuição no tempo — {name} (N={n_windows} janelas)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
        
 

def rearrange(a,y, window, overlap):
    l, f = a.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (a.itemsize*f*(window-overlap), a.itemsize*f, a.itemsize)
    X = np.lib.stride_tricks.as_strided(a, shape=shape, strides=stride)

    l,f = y.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (y.itemsize*f*(window-overlap), y.itemsize*f, y.itemsize)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=stride)
    Y = Y.max(axis=1)

    return X, Y.flatten()

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def update_stats_standardization(n, #number of samples previous standardization data 
                               mu, # mu of samples previous standardization data 
                               sd,  # std of samples previous standardization data 
                               new_batch, #  new samples
                               mode: str = "channel",            # "channel" | "timestep" | "timestep_pooled" | "window_samplewise"
                               time_cut_ratio: Optional[float] = None,  # only used for "channel"
                               eps: float = 1e-8  ):
    """
    This function combines a set of already aggregated statistics with a new batch of data.
    It uses a variation of Welford's algorithm for a numerically stable update of the variance.
    
    Args:
        n (int): Number of samples in the old data.
        mu (float or np.ndarray): Mean of the old data.
        std (float or np.ndarray): Population standard deviation of the old data.
        
    
    Returns:
        tuple: A tuple containing (n_new, mu_new, std_new), the updated statistics.
    """
    n_new_batch, mu_new_batch, sd_new_batch = get_stats_standardization(new_batch, mode=mode, time_cut_ratio= time_cut_ratio,  eps=eps)

    # new combined average
    
    n_total = n + n_new_batch
    mu_total = (n * mu + n_new_batch * mu_new_batch) / n_total

    # --- computes new standard deviation - convert std to variance 
    
    var = sd**2
    var_new_batch = sd_new_batch**2

    # M2 is the sum of square differences from mean. M2 = var * n
    m2 = var * n
    m2_new_batch = var_new_batch * n_new_batch

    # Chan equation to combine M2 of two sets
    delta = mu_new_batch - mu
    m2_total = m2 + m2_new_batch + (delta**2 * n * n_new_batch) / n_total

    var_total = m2_total / n_total
    std_total = np.sqrt(var_total)

    return n_total, mu_total, std_total




def get_stats_standardization(
    data,
    mode: str = "channel",            # "channel" | "timestep" | "timestep_pooled" | "window_samplewise"
    time_cut_ratio: Optional[float] = None,  # only used for "channel"
    eps: float = 1e-8    
) -> Tuple:
    """
    computes mean, std and number of samples 
 
    Args:
        data: data
        standardization mode to consider
        time cut ratio to mode='channel'
        eps
         
    Returns:
        tuple: (n, mu, std) computed from data
    """
    # --- get numpy views of data
    assert data.X.ndim == 3, "Expected shape (N, C, T)."
    X_tr =data.X.detach().to(dtype=torch.float32, device='cpu').numpy()
    N, C, T = X_tr.shape

    # --- mode-specific μ/σ computation (from TRAIN if not provided)
    if mode == "channel":
        # μ/σ shape: (1, C, 1), computed over (N,T)
        if time_cut_ratio is not None:
           assert 0 < time_cut_ratio <= 1.0
           cut = max(1, int(round(T * time_cut_ratio)))
           sel = slice(T - cut, T)
        else:
           sel = slice(0, T)
        mu = X_tr[:, :, sel].mean(axis=(0, 2), keepdims=True)          # (1,C,1)
        sd = X_tr[:, :, sel].std(axis=(0, 2), keepdims=True) + eps     # (1,C,1)
        
    elif mode == "timestep":
        # μ/σ shape: (1, C, T), computed over N
         mu = X_tr.mean(axis=0, keepdims=True)                           # (1,C,T)
         sd = X_tr.std(axis=0, keepdims=True) + eps
        
    elif mode == "timestep_pooled":
         mu = X_tr.mean(axis=(0, 1), keepdims=True)                      # (1,1,T)
         sd = X_tr.std(axis=(0, 1), keepdims=True) + eps
       
    elif mode == "window_samplewise":
        # Each dataset uses its own per-window stats; external mu/sd are ignored by design.
        # μ/σ shape: (N, C, 1), computed over T
        mu = X_tr.mean(axis=2, keepdims=True)                            # (N,C,1)
        sd = X_tr.std(axis=2, keepdims=True) + eps
  
    else:
        raise ValueError("mode must be 'channel', 'timestep', 'timestep_pooled', or 'window_samplewise'.")

    return (N, mu, sd)

def standarize_data(
    train_dataset,
    test_dataset=None,
    *,
    mode: str = "channel",            # "channel" | "timestep" | "timestep_pooled" | "window_samplewise"
    time_cut_ratio: Optional[float] = None,  # only used for "channel"
    eps: float = None,    
    mu=None,                          # np.ndarray or torch.Tensor (broadcastable)
    sd=None,                          # np.ndarray or torch.Tensor (broadcastable)
    apply_to_train: bool = True,      # <-- NEW: skip re-standardizing train on the 2nd call
) -> Tuple:
    """
    Standardizes data shaped (N, C, T) according to 'mode'.
    - Computes μ/σ from TRAIN if not provided, then applies to TEST (and TRAIN if apply_to_train=True).
    - For 'window_samplewise', μ/σ are per-window; external mu/sd are ignored by design.

    Returns:
        (train_dataset, test_dataset) if test_dataset is not None, else (train_dataset, )
    """
    # --- helpers
    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _set_X(dataset, X):
        dataset.X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float()

    def _store_stats(dataset, mu_arr, sd_arr):
        dataset.mu = mu_arr
        dataset.sd = sd_arr

    # --- get numpy views of data
    assert train_dataset.X.ndim == 3, "Expected shape (N, C, T)."
    X_tr = train_dataset.X.detach().to(dtype=torch.float32, device='cpu').numpy()
    N, C, T = X_tr.shape

    X_te = None
    if test_dataset is not None:
        assert test_dataset.X.ndim == 3, "Expected shape (N, C, T) for test."
        X_te = test_dataset.X.detach().to(dtype=torch.float32, device='cpu').numpy()
        Ct, Tt = X_te.shape[1], X_te.shape[2]
        if mode in ("timestep", "timestep_pooled", "channel"):
            # these modes assume aligned C,T
            assert Ct == C and Tt == T, f"Test (C,T)=({Ct},{Tt}) != Train (C,T)=({C},{T})."

    mu = _to_np(mu)
    sd = _to_np(sd)

    # --- mode-specific μ/σ computation (from TRAIN if not provided)
    if mode == "channel":
        # μ/σ shape: (1, C, 1), computed over (N,T)
        if (mu is None) or (sd is None):
            if time_cut_ratio is not None:
                assert 0 < time_cut_ratio <= 1.0
                cut = max(1, int(round(T * time_cut_ratio)))
                sel = slice(T - cut, T)
            else:
                sel = slice(0, T)
            mu = X_tr[:, :, sel].mean(axis=(0, 2), keepdims=True)          # (1,C,1)
            sd = X_tr[:, :, sel].std(axis=(0, 2), keepdims=True) + eps     # (1,C,1)
        else:
            mu = np.asarray(mu).reshape(1, C, 1)
            sd = np.asarray(sd).reshape(1, C, 1)

        # apply
        if apply_to_train:
            _set_X(train_dataset, (X_tr - mu) / sd)
        _store_stats(train_dataset, mu, sd)

        if X_te is not None:
            _set_X(test_dataset, (X_te - mu) / sd)
            _store_stats(test_dataset, mu, sd)

    elif mode == "timestep":
        # μ/σ shape: (1, C, T), computed over N
        if (mu is None) or (sd is None):
            mu = X_tr.mean(axis=0, keepdims=True)                           # (1,C,T)
            sd = X_tr.std(axis=0, keepdims=True) + eps
        else:
            mu = np.asarray(mu).reshape(1, C, T)
            sd = np.asarray(sd).reshape(1, C, T)

        if apply_to_train:
            _set_X(train_dataset, (X_tr - mu) / sd)
        _store_stats(train_dataset, mu, sd)

        if X_te is not None:
            _set_X(test_dataset, (X_te - mu) / sd)
            _store_stats(test_dataset, mu, sd)

    elif mode == "timestep_pooled":
        # μ/σ shape: (1, 1, T), computed over (N,C)
        if (mu is None) or (sd is None):
            mu = X_tr.mean(axis=(0, 1), keepdims=True)                      # (1,1,T)
            sd = X_tr.std(axis=(0, 1), keepdims=True) + eps
        else:
            mu = np.asarray(mu).reshape(1, 1, T)
            sd = np.asarray(sd).reshape(1, 1, T)

        if apply_to_train:
            _set_X(train_dataset, (X_tr - mu) / sd)
        _store_stats(train_dataset, mu, sd)

        if X_te is not None:
            _set_X(test_dataset, (X_te - mu) / sd)
            _store_stats(test_dataset, mu, sd)

    elif mode == "window_samplewise":
        # Each dataset uses its own per-window stats; external mu/sd are ignored by design.
        # μ/σ shape: (N, C, 1), computed over T
        def _win_z(X):
            mu_w = X.mean(axis=2, keepdims=True)                            # (N,C,1)
            sd_w = X.std(axis=2, keepdims=True) + eps
            return (X - mu_w) / sd_w, mu_w, sd_w

        if apply_to_train:
            X_tr_std, mu_tr, sd_tr = _win_z(X_tr)
            _set_X(train_dataset, X_tr_std)
            _store_stats(train_dataset, mu_tr, sd_tr)
        else:
            # still compute & store stats in case the caller wants them
            mu_tr = X_tr.mean(axis=2, keepdims=True)
            sd_tr = X_tr.std(axis=2, keepdims=True) + eps
            _store_stats(train_dataset, mu_tr, sd_tr)

        if X_te is not None:
            X_te_std, mu_te, sd_te = _win_z(X_te)
            _set_X(test_dataset, X_te_std)
            _store_stats(test_dataset, mu_te, sd_te)

    else:
        raise ValueError("mode must be 'channel', 'timestep', 'timestep_pooled', or 'window_samplewise'.")

    return (train_dataset, test_dataset) if test_dataset is not None else (train_dataset,)

    

       
       
       
def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None



def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()
        
def concat_samples(datasetDst, datasetSample):
     
     datasetDst.X = torch.cat((datasetDst.X, datasetSample.X),0)
     datasetDst.Y = torch.cat((datasetDst.Y, datasetSample.Y),0)

     return datasetDst
 
    
 
def find_column_index(csv_file, column_name):
    with open(csv_file, 'r', newline='') as csv_file:
        # Set the space as the delimiter for the reader
        reader = csv.reader(csv_file, delimiter=' ')
        header_row = next(reader)
        try:
            column_index = header_row.index(column_name)
            return column_index
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

def extract_column_to_txt(csv_file, column_name, txt_file, column_type):
    column_index = find_column_index(csv_file, column_name)

    with open(csv_file, 'r', newline='') as csv_file:
        # Set the space as the delimiter for the reader
        reader = csv.reader(csv_file, delimiter=' ')
        next(reader)
        column_data = [row[column_index] for row in reader]
        
        
    with open(txt_file, 'w') as txt_file:
         lines = []
         for value in column_data:
             if column_type == 'int':
                lines.append(str(int(float(value))))
             else:
                lines.append(value)
         txt_file.write('\n'.join(lines))
'''
    with open(txt_file, 'w') as txt_file:
        for value in column_data:
            if column_type == 'int':
               txt_file.write(str(int(float(value))) + '\n')
            else:
               txt_file.write(value + '\n') 
'''


def create_directory (path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass


def remove_columns(csv_file, columns, txt_file):
    df = pd.read_csv(csv_file, delimiter=' ')
    df = df.drop(columns=columns)
    df.to_csv(txt_file, header=False,index=False, sep=' ')
            
            
def delete_file(file_path):
    path = Path(file_path)
    path.unlink()            
                
def generate_filtered_csv(csv_file, column_name, filter_values, csv_output_file):
    with open(csv_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        header_row = next(reader)
        
        #processed_header = header_row[1:]
        
        # Find the index of the specified column
        try:
            column_index = header_row.index(column_name)
            #column_index = processed_header.index(column_name) + 1  # Shift index due to removed first column
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        # Filter rows based on the values of the specified column
        filtered_rows = [row for row in reader if row[column_index] in filter_values]
        #filtered_rows = [row[1:] for row in reader if row[column_index] in filter_values]

    with open(csv_output_file, 'w', newline='') as csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=' ')

        # Write the header row
        writer.writerow(header_row)
        #writer.writerow(processed_header)

        # Write the filtered rows
        writer.writerows(filtered_rows)

# Jittering
# "Jittering" can be considered as "applying different noise to each sample".
# sigma = standard devitation (STD) of the noise
# source https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
def DA_Jitter(X, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)

    return X+noise

# scaling
#"Scaling" can be considered as "applying constant noise to the entire samples"
# sigma = STD of the zoom-in/out factor
# adapted from https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[1],1))
    X[:,:,:] = X[:,:,:] * scalingFactor[:]

    return X

#source : https://github.com/Human-Signals-Lab/LAPNet-HAR
def DA_MagWarp(x, sigma=0.2, knot=4):

    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

#source : https://github.com/Human-Signals-Lab/LAPNet-HAR

def DA_TimeWarp(x, sigma=0.2, knot=4):
  
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

'''
# Rotation
def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))
'''
# Permutation
# adapted from https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
#### Hyperparameters :  nPerm = # of segments to permute
#### minSegLength = allowable minimum length for each segment


def DA_Permutation(X, Y, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    Y_new = np.zeros(X.shape[0], dtype=int)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        y_temp = Y[segs[idx[ii]]:segs[idx[ii]+1]]
        X_new[pp:pp+len(x_temp),:] = x_temp
        Y_new[pp:pp+len(x_temp)] = y_temp
        pp += len(x_temp)
    return(X_new, Y_new)



def augmentation_instance(X, data_augmentation):
    
    data = copy.deepcopy(X)   
  
    
    if 'Jitter' in data_augmentation: # Jitter
         print('JITTER')
         X_temp = DA_Jitter(data, sigma=0.05)
             
    if 'Scale' in  data_augmentation:   # Scale
         print('SCALE')
         X_temp = DA_Scaling(data, sigma=0.1)
        
    if 'Perm' in data_augmentation:   # Permutation
         print('PERM')
         X_temp = DA_Permutation(data,data.Y, nPerm=4, minSegLength=10)           
 
    if 'TimeW' in data_augmentation: # TimeWarping  
         print('TIMEWARPING')
         X_temp = DA_TimeWarp(data, sigma=0.2, knot=4)           
         
    if 'MagW' in data_augmentation: # Magnetude Warping 
         print('MAGWARPING')
         X_temp = DA_MagWarp(data, sigma=0.2, knot=4)           
         
    return X_temp  

