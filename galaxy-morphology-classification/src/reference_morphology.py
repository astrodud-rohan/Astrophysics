"""
Reference parameters anchoring the synthetic galaxy image simulation to
real, published values.
 
Sources:
    - Sersic profile: de Vaucouleurs (1948) n=4 profile for ellipticals;
      exponential disk n=1 for spiral disks (standard galaxy structural
      decomposition, e.g. Simard et al. 2011 SDSS bulge-disk decomposition)
    - SDSS imaging characteristics: ~0.396 arcsec/pixel plate scale,
      ~1.4 arcsec median seeing (PSF FWHM), gri-band sky background
      surface brightness ~21-22 mag/arcsec^2 (SDSS imaging survey
      technical papers, York et al. 2000; Stoughton et al. 2002)
    - Galaxy Zoo morphology classes and rough class incidence in the
      local universe (spiral ~60%, elliptical ~25%, merger/irregular
      ~5-10% in typical SDSS-depth samples) -- Lintott et al. 2008,
      2011 Galaxy Zoo papers
    - CAS morphology statistics (Concentration, Asymmetry, Smoothness):
      Conselice 2003; Gini/M20: Lotz et al. 2004 -- the actual
      pre-deep-learning quantitative morphology feature set used in
      the literature, reproduced here as the "classical" baseline
      feature set
    - Spiral arm pitch angle: typical range ~10-30 degrees for
      logarithmic spiral galaxies (Kennicutt 1981 pitch angle survey)
 
This file is the single documented source of "real-world anchoring."
All downstream simulation code samples nuisance parameters (position
angle, exact pitch angle, noise realization) around these anchors.
"""
 
SERSIC_INDEX = dict(elliptical=4.0, spiral_disk=1.0, spiral_bulge=2.5)
 
SDSS_IMAGING = dict(
    pixel_scale_arcsec=0.396,
    psf_fwhm_arcsec=1.4,
    sky_mag_per_arcsec2=21.5,   # approximate gri-band sky brightness
    zeropoint_mag=22.5,          # approximate SDSS zeropoint for a simple mag<->counts conversion
    image_size_px=64,            # cutout size used for this project (real SDSS cutouts are often larger)
)
 
GALAXY_ZOO_CLASS_FRACTIONS = dict(spiral=0.55, elliptical=0.30, merger=0.15)
 
SPIRAL_PITCH_ANGLE_DEG = (10, 30)  # (min, max) typical range
 
MERGER_SEPARATION_RANGE_PX = (6, 20)  # projected nuclei separation in simulated cutouts
