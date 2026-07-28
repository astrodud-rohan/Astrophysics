"""
Matched-filter search: run a bank of IMRPhenomD templates (varying
component masses) against the injected strain, find the best-matching
template and peak SNR/time -- mirroring a real detection-pipeline
search stage (e.g. PyCBC's own production search, simplified to a
single-detector, non-spinning grid for this project's scope).
"""
import numpy as np
import pandas as pd
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform
 
from reference_events import GW150914_PARAMS, F_LOWER_HZ, SAMPLE_RATE_HZ
 
DATA_PATH = "data/injected_strain.npz"
OUT_PATH = "data/search_results.csv"
 
 
def load_strain():
    d = np.load(DATA_PATH)
    delta_t = float(d["delta_t"])
    strain = TimeSeries(d["strain"], delta_t=delta_t)
    psd = FrequencySeries(d["psd_data"], delta_f=float(d["psd_delta_f"]))
    return strain, psd, d
 
 
def search_template(strain, psd, mass1, mass2, distance=1000.0):
    delta_t = strain.delta_t
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD", mass1=mass1, mass2=mass2,
        distance=distance, delta_t=delta_t, f_lower=F_LOWER_HZ,
    )
    hp.resize(len(strain))
    template = hp.cyclic_time_shift(hp.start_time)
    snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=F_LOWER_HZ)
    snr = snr.crop(4, 4)
    peak_idx = np.argmax(np.abs(snr.data))
    return abs(snr.data[peak_idx]), snr.sample_times[peak_idx]
 
 
def run_template_bank(strain, psd, n_grid=15):
    """Grid over (mass1, mass2) around the true injected values --
    equivalent in spirit to a real chirp-mass/mass-ratio template bank,
    simplified to a direct component-mass grid for clarity."""
    m1_grid = np.linspace(15, 55, n_grid)
    m2_grid = np.linspace(15, 45, n_grid)
 
    rows = []
    for m1 in m1_grid:
        for m2 in m2_grid:
            if m2 > m1:
                continue  # convention: mass1 >= mass2
            try:
                peak_snr, peak_time = search_template(strain, psd, m1, m2)
            except Exception:
                continue
            chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
            rows.append(dict(mass1=m1, mass2=m2, chirp_mass=chirp_mass,
                              peak_snr=peak_snr, peak_time=peak_time))
    return pd.DataFrame(rows)
 
 
def main():
    strain, psd, d = load_strain()
    print("Running template bank search...")
    results = run_template_bank(strain, psd, n_grid=15)
    results.to_csv(OUT_PATH, index=False)
 
    best = results.loc[results["peak_snr"].idxmax()]
    p = GW150914_PARAMS
    true_chirp_mass = (p["mass1_source"] * p["mass2_source"]) ** 0.6 / \
        (p["mass1_source"] + p["mass2_source"]) ** 0.2
 
    print(f"\nBest-matching template: mass1={best.mass1:.1f}, mass2={best.mass2:.1f}, "
          f"chirp_mass={best.chirp_mass:.2f} Msun, SNR={best.peak_snr:.2f}, time={best.peak_time:.3f}s")
    print(f"True injected: mass1={p['mass1_source']}, mass2={p['mass2_source']}, "
          f"chirp_mass={true_chirp_mass:.2f} Msun")
    print(f"True merger time: {float(d['true_merger_time_s']):.3f}s")
    print(f"\nSaved full grid -> {OUT_PATH}")
 
 
if __name__ == "__main__":
    main()