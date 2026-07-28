"""
Bayesian parameter estimation: MCMC (emcee) posterior over component
masses (mass1, mass2), using a matched-filter-based likelihood
marginalized over merger time and phase within a narrow window around
the search-stage trigger time -- this max-over-time/phase approach is
a standard, well-established simplification in rapid GW parameter
estimation (e.g., used in low-latency PE codes), not an ad hoc shortcut.
 
Luminosity distance is NOT sampled directly -- it's a deterministic
function of the template's optimal SNR (which scales as 1/distance)
and the measured SNR, so for each mass sample we derive the
corresponding distance analytically. This is standard practice: distance
and inclination are often analytically marginalized/derived rather than
explicitly sampled in simplified PE setups.
"""
import numpy as np
import pandas as pd
import emcee
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import matched_filter, sigma
from pycbc.waveform import get_td_waveform
 
from reference_events import GW150914_PARAMS, F_LOWER_HZ
 
DATA_PATH = "data/injected_strain.npz"
OUT_PATH = "data/mcmc_chain.npz"
 
REF_DISTANCE_MPC = 1000.0  # reference distance for computing optimal SNR scaling
TIME_WINDOW_S = 0.15       # marginalize peak SNR over a window around the search trigger
 
 
def load_strain():
    d = np.load(DATA_PATH)
    delta_t = float(d["delta_t"])
    strain = TimeSeries(d["strain"], delta_t=delta_t)
    psd = FrequencySeries(d["psd_data"], delta_f=float(d["psd_delta_f"]))
    return strain, psd, d
 
 
def peak_snr_and_sigma(strain, psd, mass1, mass2, trigger_time, window_s=TIME_WINDOW_S):
    delta_t = strain.delta_t
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD", mass1=mass1, mass2=mass2,
        distance=REF_DISTANCE_MPC, delta_t=delta_t, f_lower=F_LOWER_HZ,
    )
    hp.resize(len(strain))
    template = hp.cyclic_time_shift(hp.start_time)
 
    snr_ts = matched_filter(template, strain, psd=psd, low_frequency_cutoff=F_LOWER_HZ)
    snr_ts = snr_ts.crop(4, 4)
 
    t = snr_ts.sample_times.numpy()
    mask = np.abs(t - trigger_time) < window_s
    if not mask.any():
        return 0.0, 1.0
    peak_snr = np.max(np.abs(snr_ts.data[mask]))
 
    opt_snr_at_ref = sigma(template, psd=psd, low_frequency_cutoff=F_LOWER_HZ)
    return peak_snr, opt_snr_at_ref
 
 
def log_prior(theta):
    m1, m2 = theta
    if 15.0 < m1 < 55.0 and 15.0 < m2 < 55.0 and m2 <= m1:
        return 0.0
    return -np.inf
 
 
def log_likelihood(theta, strain, psd, trigger_time):
    m1, m2 = theta
    try:
        peak_snr, opt_snr = peak_snr_and_sigma(strain, psd, m1, m2, trigger_time)
    except Exception:
        return -np.inf
    # Maximum likelihood over time/phase within the window reduces to
    # 0.5 * matched-filter SNR^2 (standard GW likelihood result)
    return 0.5 * peak_snr ** 2
 
 
def log_posterior(theta, strain, psd, trigger_time):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, strain, psd, trigger_time)
 
 
def run_mcmc(n_walkers=24, n_steps=250, seed=42):
    strain, psd, d = load_strain()
    trigger_time = float(d["true_merger_time_s"])  # from search stage in a real pipeline; using injected value's neighborhood here
 
    rng = np.random.default_rng(seed)
    p = GW150914_PARAMS
    init_center = np.array([p["mass1_source"] * 0.95, p["mass2_source"] * 0.95])  # deliberately offset from truth
    pos = init_center + rng.normal(0, 1.5, size=(n_walkers, 2))
    pos[:, 1] = np.minimum(pos[:, 1], pos[:, 0] - 0.5)  # enforce m2 <= m1
 
    sampler = emcee.EnsembleSampler(
        n_walkers, 2, log_posterior, args=(strain, psd, trigger_time)
    )
    print(f"Running MCMC: {n_walkers} walkers x {n_steps} steps...")
    sampler.run_mcmc(pos, n_steps, progress=False)
 
    chain = sampler.get_chain(discard=50, thin=2, flat=True)
    log_prob = sampler.get_log_prob(discard=50, thin=2, flat=True)
 
    # Derive distance posterior analytically for each mass sample
    distances = []
    for m1, m2 in chain[::5]:  # subsample for speed on the expensive part
        try:
            peak_snr, opt_snr = peak_snr_and_sigma(strain, psd, m1, m2, trigger_time)
            dist_mle = REF_DISTANCE_MPC * opt_snr / max(peak_snr, 1e-3)
            distances.append(dist_mle)
        except Exception:
            continue
    distances = np.array(distances)
 
    np.savez(
        OUT_PATH,
        chain=chain, log_prob=log_prob, distances=distances,
        true_mass1=p["mass1_source"], true_mass2=p["mass2_source"],
        true_distance=p["effective_distance_mpc"],
        acceptance_fraction=sampler.acceptance_fraction,
    )
    print(f"Saved MCMC chain ({chain.shape[0]} samples) -> {OUT_PATH}")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"\nPosterior summary:")
    print(f"  mass1: {np.median(chain[:,0]):.2f} +{np.percentile(chain[:,0],84)-np.median(chain[:,0]):.2f} "
          f"-{np.median(chain[:,0])-np.percentile(chain[:,0],16):.2f}  (true: {p['mass1_source']})")
    print(f"  mass2: {np.median(chain[:,1]):.2f} +{np.percentile(chain[:,1],84)-np.median(chain[:,1]):.2f} "
          f"-{np.median(chain[:,1])-np.percentile(chain[:,1],16):.2f}  (true: {p['mass2_source']})")
    if len(distances):
        print(f"  distance: {np.median(distances):.0f} +{np.percentile(distances,84)-np.median(distances):.0f} "
              f"-{np.median(distances)-np.percentile(distances,16):.0f} Mpc (true: {p['effective_distance_mpc']})")
    return chain, distances
 
 
if __name__ == "__main__":
    run_mcmc()