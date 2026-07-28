"""
Build a realistic "detector strain" dataset: a real IMRPhenomD waveform
(pycbc, research-grade) injected into a colored Gaussian noise
realization drawn from the real aLIGO design-sensitivity PSD.
 
DISCLOSURE: strain is SIMULATED (injected signal + noise realization),
not downloaded real detector data (see reference_events.py). 
Waveform physics and noise PSD are real, research-grade (pycbc / published aLIGO curves).
"""
import numpy as np
import pycbc.noise
import pycbc.psd
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
 
from reference_events import GW150914_PARAMS, F_LOWER_HZ, SAMPLE_RATE_HZ, SEGMENT_DURATION_S
 
OUT_PATH = "data/injected_strain.npz"
 
 
def make_signal(mass1, mass2, distance_mpc, delta_t, f_lower, spin1z=0.0, spin2z=0.0):
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=mass1, mass2=mass2,
        spin1z=spin1z, spin2z=spin2z,
        distance=distance_mpc,
        delta_t=delta_t, f_lower=f_lower,
    )
    return hp  # plus polarization, face-on simplification
 
 
def build_dataset(seed=42):
    p = GW150914_PARAMS
    delta_t = 1.0 / SAMPLE_RATE_HZ
    n_samples = int(SEGMENT_DURATION_S * SAMPLE_RATE_HZ)
    delta_f = 1.0 / SEGMENT_DURATION_S
    flen = n_samples // 2 + 1
 
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, F_LOWER_HZ)
 
    rng_seed = seed
    noise = pycbc.noise.noise_from_psd(n_samples, delta_t, psd, seed=rng_seed)
 
    hp = make_signal(
        p["mass1_source"], p["mass2_source"], p["effective_distance_mpc"],
        delta_t, F_LOWER_HZ, p["spin1z"], p["spin2z"],
    )
 
    # Place merger near the end of the segment. IMPORTANT: the LAST
    # sample of the hp array is NOT the merger -- get_td_waveform pads
    # the array (to a convenient FFT length) with a trailing buffer
    # after the actual coalescence/ringdown. The true merger corresponds
    # to the sample of PEAK AMPLITUDE within hp. We anchor on that peak
    # directly so our recorded "true merger time" matches what a matched
    # filter search will actually recover.
    strain = noise.copy()
    hp_samples = len(hp)
    peak_index_in_hp = int(np.argmax(np.abs(hp.data)))
 
    target_peak_sample = n_samples - int(8.0 * SAMPLE_RATE_HZ)  # merger lands 8s before segment end (room for trailing pad)
    start = target_peak_sample - peak_index_in_hp
    if start < 0:
        raise ValueError(
            f"Segment too short to hold full waveform before the target "
            f"peak placement. Increase SEGMENT_DURATION_S."
        )
    end = start + hp_samples
    if end > n_samples:
        raise ValueError(
            f"Segment too short: waveform tail extends past segment end. "
            f"Increase SEGMENT_DURATION_S."
        )
    strain.data[start:end] += hp.data
    true_merger_sample = start + peak_index_in_hp
 
    np.savez(
        OUT_PATH,
        strain=np.array(strain.data),
        noise_only=np.array(noise.data),
        signal_only_padded=np.pad(np.array(hp.data), (start, n_samples - end), mode="constant")[:n_samples],
        delta_t=delta_t,
        psd_data=np.array(psd.data),
        psd_delta_f=delta_f,
        injected_params=np.array([p["mass1_source"], p["mass2_source"], p["effective_distance_mpc"]]),
        true_merger_sample=true_merger_sample,
        true_merger_time_s=true_merger_sample * delta_t,
    )
    print(f"Saved injected strain ({n_samples} samples, {SEGMENT_DURATION_S}s @ {SAMPLE_RATE_HZ}Hz) -> {OUT_PATH}")
    return strain, noise, hp
 
 
if __name__ == "__main__":
    build_dataset()
