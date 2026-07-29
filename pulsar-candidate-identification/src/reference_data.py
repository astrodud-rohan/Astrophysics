"""
Reference / provenance documentation for the REAL HTRU2 dataset used in
this project.
 
DATA SOURCE (real, not synthetic):
    GitHub IS reachable, and the dataset creator himself,
    Dr. Rob Lyon (University of Manchester, Jodrell Bank Centre for
    Astrophysics), hosts the original HTRU2 data in his own public
    repository:
        https://github.com/scienceguyrob/PulsarFeatureLab
    fetched here via `codeload.github.com` (repo tarball download).
    This is the authoritative primary source -- the same data
    distributed via the UCI ML Repository and Kaggle mirrors.
 
CITATION:
    R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles,
    "Fifty Years of Pulsar Candidate Selection: From simple filters to
    a new principled real-time classification approach", Monthly
    Notices of the Royal Astronomical Society, 459 (1), 1104-1123, 2016.
    DOI: 10.1093/mnras/stw656
    Dataset DOI: 10.6084/m9.figshare.3080389.v1
 
SURVEY: High Time Resolution Universe Survey (South) -- HTRU-S,
    a real pulsar survey conducted with the Parkes radio telescope.
 
DATASET COMPOSITION (verified against documentation):
    17,898 total candidates
    1,639 real pulsars (human-annotator-confirmed)
    16,259 non-pulsar / RFI / noise candidates
    -> a genuine, severe class imbalance (~9.9:1), which is itself the
    central real-world challenge in pulsar survey classification.
 
FEATURES (8, "Lyon et al. 2015" feature set -- the actual published
    feature set, extracted via Lyon's own PulsarFeatureLab tool from
    each candidate's folded pulse profile and DM-SNR curve):
    1. Mean of the integrated (folded) pulse profile
    2. Standard deviation of the integrated pulse profile
    3. Excess kurtosis of the integrated pulse profile
    4. Skewness of the integrated pulse profile
    5. Mean of the DM-SNR curve
    6. Standard deviation of the DM-SNR curve
    7. Excess kurtosis of the DM-SNR curve
    8. Skewness of the DM-SNR curve
    (class label: 0 = non-pulsar/RFI, 1 = real pulsar)
 
A 30-feature "combined" variant (Lyon 2015's 8 features + Thornton et
al. 2013's 22 additional features) is also available in this repo and
used for a secondary richer-feature-set comparison.
"""
 
RAW_LYON8_PATH = "../data/raw/HTRU_2_Lyon_Features_8.csv"
RAW_COMBINED30_PATH = "../data/raw/HTRU_2_Combined_30.csv"
 
LYON8_COLUMNS = [
    "mean_ip", "std_ip", "kurtosis_ip", "skew_ip",
    "mean_dmsnr", "std_dmsnr", "kurtosis_dmsnr", "skew_dmsnr",
    "class",
]
 
N_TOTAL_DOCUMENTED = 17898
N_PULSAR_DOCUMENTED = 1639
N_NONPULSAR_DOCUMENTED = 16259
