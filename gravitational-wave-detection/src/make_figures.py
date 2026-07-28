import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform
 
from reference_events import GW150914_PARAMS, F_LOWER_HZ
 
DATA_DIR = "data"
FIG_DIR = "figures"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
 
d = np.load(f"{DATA_DIR}/injected_strain.npz")
delta_t = float(d["delta_t"])
strain = TimeSeries(d["strain"], delta_t=delta_t)
psd = FrequencySeries(d["psd_data"], delta_f=float(d["psd_delta_f"]))
signal_only = d["signal_only_padded"]
true_merger_time = float(d["true_merger_time_s"])
p = GW150914_PARAMS
 
# ---------- 1. Raw strain around the merger ----------
t = np.arange(len(strain.data)) * delta_t
window = (t > true_merger_time - 0.5) & (t < true_merger_time + 0.1)
 
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t[window], strain.data[window], color="#4a5568", lw=0.7)
axes[0].set_title("Simulated detector strain (signal + colored aLIGO noise)")
axes[0].set_ylabel("Strain")
 
axes[1].plot(t[window], signal_only[window], color="#c53030", lw=1.2)
axes[1].set_title("Injected signal only (IMRPhenomD waveform, mass1=36.2, mass2=29.1 Msun)")
axes[1].set_ylabel("Strain")
axes[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_strain_and_signal.png")
plt.close()
 
# ---------- 2. Matched filter SNR timeseries ----------
hp, hc = get_td_waveform(approximant="IMRPhenomD", mass1=p["mass1_source"], mass2=p["mass2_source"],
                          distance=p["effective_distance_mpc"], delta_t=delta_t, f_lower=F_LOWER_HZ)
hp.resize(len(strain))
template = hp.cyclic_time_shift(hp.start_time)
snr_ts = matched_filter(template, strain, psd=psd, low_frequency_cutoff=F_LOWER_HZ)
snr_ts = snr_ts.crop(4, 4)
snr_t = snr_ts.sample_times.numpy()
snr_abs = np.abs(snr_ts.data)
 
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(snr_t, snr_abs, color="#2b6cb0", lw=0.8)
ax.axvline(true_merger_time, color="#c53030", linestyle="--", lw=1, label=f"true merger t={true_merger_time:.2f}s")
peak_idx = np.argmax(snr_abs)
ax.plot(snr_t[peak_idx], snr_abs[peak_idx], "o", color="#c53030", ms=8,
        label=f"recovered peak SNR={snr_abs[peak_idx]:.1f}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("|Matched filter SNR|")
ax.set_title("Matched-filter SNR time series (true-parameter template)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_snr_timeseries.png")
plt.close()
 
# ---------- 3. Template bank search heatmap ----------
search = pd.read_csv(f"{DATA_DIR}/search_results.csv")
pivot = search.pivot_table(index="mass2", columns="mass1", values="peak_snr")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap="viridis", shading="auto")
ax.scatter([p["mass1_source"]], [p["mass2_source"]], color="red", marker="*", s=200,
           label="true injected values", zorder=5)
ax.set_xlabel("mass1 (Msun)")
ax.set_ylabel("mass2 (Msun)")
ax.set_title("Template bank matched-filter search: peak SNR over (mass1, mass2) grid")
plt.colorbar(im, ax=ax, label="Peak SNR")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_template_bank_search.png")
plt.close()
 
# ---------- 4. Corner plot: MCMC posterior on masses ----------
mcmc = np.load(f"{DATA_DIR}/mcmc_chain.npz")
chain = mcmc["chain"]
fig = corner.corner(
    chain, labels=["mass1 (Msun)", "mass2 (Msun)"],
    truths=[p["mass1_source"], p["mass2_source"]],
    show_titles=True, title_fmt=".2f", color="#2b6cb0",
    truth_color="#c53030",
)
fig.suptitle("MCMC posterior: component masses", y=1.02)
fig.savefig(f"{FIG_DIR}/04_mass_posterior_corner.png", bbox_inches="tight")
plt.close(fig)
 
# ---------- 5. Distance posterior ----------
distances = mcmc["distances"]
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(distances, bins=30, color="#38a169", alpha=0.75, density=True)
ax.axvline(p["effective_distance_mpc"], color="#c53030", linestyle="--", lw=2,
           label=f"true effective distance = {p['effective_distance_mpc']:.0f} Mpc")
ax.set_xlabel("Luminosity distance (Mpc)")
ax.set_ylabel("Posterior density")
ax.set_title("Derived distance posterior (from SNR-distance scaling)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_distance_posterior.png")
plt.close()
 
print("All figures saved to", FIG_DIR)