"""
Reference parameters for real, published gravitational wave events.
 
DISCLOSURE: This project does NOT download real LIGO strain data. 
HOWEVER, the physics tooling here IS the real research-grade software: `pycbc` generates actual
IMRPhenomD waveform templates (the same approximant family used in
real LIGO/Virgo parameter estimation), and the detector noise is drawn
from PyCBC's built-in analytic aLIGO design-sensitivity PSD curves
(the same published noise curves used in real sensitivity studies).
What's synthetic is specifically the STRAIN DATA ITSELF (we inject a
known signal into a colored noise realization rather than downloading
GW150914's actual recorded strain) -- everything else in the pipeline
(waveform physics, noise statistics, matched filtering, Bayesian
inference machinery) is the same as a real analysis.
 
Source for parameter values below: Abbott et al. 2016, "Observation of
Gravitational Waves from a Binary Black Hole Merger" (GW150914
discovery paper, Phys. Rev. Lett. 116, 061102), values as commonly
cited (recalled from training knowledge, not live-queried).
"""
 
GW150914_PARAMS = dict(
    mass1_source=36.2,       # solar masses, primary (source frame)
    mass2_source=29.1,       # solar masses, secondary (source frame)
    final_mass_source=62.3,  # solar masses, remnant black hole
    distance_mpc=410.0,      # Mpc, PUBLISHED luminosity distance (median)
    # Effective distance used for THIS injection: aLIGOZeroDetHighPower is
    # a late-observing-run DESIGN sensitivity curve, which is quieter
    # (better sensitivity) than the actual O1-era detector noise present
    # when GW150914 was really observed. Using the real 410 Mpc against
    # this quieter design PSD yields an unrealistically high SNR (~96)
    # that doesn't match the real reported detection. To keep the
    # matched-filter SNR and parameter-estimation posterior widths
    # realistic and comparable to the real published detection
    # significance, we inject at an EFFECTIVE distance of ~1639 Mpc,
    # calibrated (via pycbc.filter.sigma) to reproduce the real reported
    # network SNR of ~24 under this specific PSD. This is a deliberate,
    # documented modeling choice, not the literal source distance.
    effective_distance_mpc=1639.0,
    spin1z=0.0,              # low/unconstrained spin, simplifying to non-spinning
    spin2z=0.0,
    inclination=0.0,         # simplified face-on for a clean detection SNR
    network_snr=23.7,        # approximate combined H1+L1 SNR reported at discovery
    merger_gps_time=1126259462.4,
)
 
# aLIGO design sensitivity, low-frequency cutoff used in real O1-era analyses
F_LOWER_HZ = 20.0
SAMPLE_RATE_HZ = 4096.0
SEGMENT_DURATION_S = 32.0  # long enough to hold the full f_lower=20Hz inspiral + ringdown + buffer
