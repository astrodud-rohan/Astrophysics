"""
Detrending + BLS (Box Least Squares) period search + feature extraction.
 
This is the classical detection stage that any candidate would run
BEFORE machine learning ever gets involved -- mirrors the real Kepler/
TESS pipeline order: detrend -> BLS search -> fold -> vet.
 
Uses astropy.timeseries.BoxLeastSquares (the same implementation used
in real research pipelines).
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from astropy.timeseries import BoxLeastSquares
 
DATA_PATH = "../data/lightcurves.npz"
META_PATH = "../data/metadata.csv"
OUT_FEATURES = "../data/features.csv"
 
 
def detrend(flux, window_days, cadence_days, poly=2):
    """Savitzky-Golay detrending, standard for removing stellar
    variability while preserving transit-shaped dips (short relative
    to the smoothing window)."""
    window_pts = int(window_days / cadence_days)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(window_pts, poly + 2)
    trend = savgol_filter(flux, window_length=window_pts, polyorder=poly, mode="interp")
    return flux / trend
 
 
def run_bls(t, flux_detrended, min_period=0.5, max_period=20.0, n_periods=1200):
    durations = np.linspace(0.02, 0.3, 5)  # days, plausible transit durations
    periods = np.linspace(min_period, max_period, n_periods)
    bls = BoxLeastSquares(t, flux_detrended)
    result = bls.power(periods, durations)
    best_idx = np.argmax(result.power)
    best_period = result.period[best_idx]
    best_duration = result.duration[best_idx]
    best_t0 = result.transit_time[best_idx]
    best_power = result.power[best_idx]
 
    # Compute the folded model stats at best period for feature extraction
    stats = bls.compute_stats(best_period, best_duration, best_t0)
    depth_val, depth_err = stats["depth"]
    depth_odd_val, _ = stats["depth_odd"]
    depth_even_val, _ = stats["depth_even"]
    return dict(
        bls_period=best_period,
        bls_duration=best_duration,
        bls_t0=best_t0,
        bls_power=best_power,
        bls_depth=depth_val,
        bls_depth_err=depth_err,
        bls_snr=depth_val / max(depth_err, 1e-12),
        bls_odd_even_mismatch=abs(depth_odd_val - depth_even_val) / max(depth_val, 1e-12),
        transit_times=stats["transit_times"],
    )
 
 
def fold_and_bin(t, flux, period, t0, n_bins=64, phase_window=0.15):
    """Fold light curve on best period, bin into a fixed-length vector
    for use as CNN input (this is the 'AstroNet'-style global view)."""
    phase = ((t - t0 + period / 2) % period) / period - 0.5
    mask = np.abs(phase) < phase_window
    order = np.argsort(phase[mask])
    p, f = phase[mask][order], flux[mask][order]
 
    bins = np.linspace(-phase_window, phase_window, n_bins + 1)
    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (p >= bins[i]) & (p < bins[i + 1])
        if sel.sum() > 0:
            binned[i] = np.median(f[sel])
    # fill any empty bins by linear interpolation
    nanmask = np.isnan(binned)
    if nanmask.any() and not nanmask.all():
        binned[nanmask] = np.interp(
            np.flatnonzero(nanmask), np.flatnonzero(~nanmask), binned[~nanmask]
        )
    elif nanmask.all():
        binned[:] = 1.0
    return binned
 
 
def main():
    npz = np.load(DATA_PATH, allow_pickle=True)
    t_full = npz["time"]
    flux_all = npz["flux"]
    ids = npz["ids"]
    meta = pd.read_csv(META_PATH)
    cadence_days = np.median(np.diff(t_full))
 
    feature_rows = []
    folded_curves = []
 
    for i in range(flux_all.shape[0]):
        flux = flux_all[i]
        flux_dt = detrend(flux, window_days=2.0, cadence_days=cadence_days)
 
        try:
            bls_res = run_bls(t_full, flux_dt)
        except Exception as e:
            bls_res = dict(
                bls_period=np.nan, bls_duration=np.nan, bls_t0=np.nan,
                bls_power=np.nan, bls_depth=np.nan, bls_depth_err=np.nan,
                bls_snr=np.nan, bls_odd_even_mismatch=np.nan, transit_times=None,
            )
 
        n_transits = (
            len(bls_res["transit_times"]) if bls_res.get("transit_times") is not None else 0
        )
 
        folded = fold_and_bin(
            t_full, flux_dt,
            bls_res["bls_period"] if not np.isnan(bls_res["bls_period"]) else 5.0,
            bls_res["bls_t0"] if not np.isnan(bls_res["bls_t0"]) else 0.0,
        )
        folded_curves.append(folded)
 
        row = {
            "id": ids[i],
            "n_transits_found": n_transits,
            **{k: v for k, v in bls_res.items() if k != "transit_times"},
        }
        feature_rows.append(row)
 
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{flux_all.shape[0]}")
 
    feat_df = pd.DataFrame(feature_rows)
    full_df = meta.merge(feat_df, on="id")
    full_df.to_csv(OUT_FEATURES, index=False)
 
    folded_curves = np.array(folded_curves)
    np.savez_compressed(
        "../data/folded_curves.npz",
        folded=folded_curves, ids=ids,
    )
 
    print(f"Saved features -> {OUT_FEATURES}")
    print(full_df[["label", "bls_snr", "bls_odd_even_mismatch"]].groupby("label").median())
 
 
if __name__ == "__main__":
    main()
