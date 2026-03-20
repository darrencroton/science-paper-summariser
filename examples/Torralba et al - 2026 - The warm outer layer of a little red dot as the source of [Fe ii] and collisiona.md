# The warm outer layer of a little red dot as the source of [Fe ii] and collisional Balmer lines with scattering wings

Authors: Torralba A., Matthee J., Pezzulli G., Naidu R. P., Ishikawa Y., Brammer G. B., Chang S.-J., Chisholm J., de Graaff A., D’Eugenio F., Di Cesare C., Eilers A.-C., Greene J. E., Gronke M., Iani E., Kokorev V., Kotiwale G., Kramarenko I., Ma Y., Mascia S., Navarrete B., Nelson E., Oesch P., Simcoe R. A., Wuyts S.

Published: February 2026 ([Link](https://arxiv.org/abs/2510.00103))

## Key Ideas

- The paper analyses deep **JWST/NIRSpec** PRISM and G395H spectra of the luminous **little red dot** (LRD) FRESCO-GN-9771 at $z=5.5$, finding a strong Balmer break, broad Balmer lines, and very narrow $[\mathrm{O\,III}]$ emission.[^1]  
- A “forest” of optical **$[\mathrm{Fe\,II}]$** lines is attributed to a dense warm layer with $n_{\mathrm{H}}=10^{9\text{--}10}\,\mathrm{cm^{-3}}$ and $T_{\mathrm{e}}\approx 7000\,\mathrm{K}$, which also produces exponential Balmer wings via electron scattering.[^2]  
- The extreme Balmer decrements (e.g. $H\alpha:H\beta:H\gamma\approx 10.4:1:0.14$) are interpreted as evidence for collisional excitation and resonant scattering dominating Balmer emission, implying Balmer lines do *not* directly trace SMBH-region kinematics and virial BH masses are likely overestimated.[^3]  

## Introduction

- JWST has revealed “little red dots (LRDs)” as “faint compact broad Balmer line emitters” at $z=2\text{--}9$, often with red UV-to-optical colours “due to a Balmer break”.[^4]  
- The powering mechanism is debated, but LRDs are “primarily thought to be driven by accretion onto supermassive black holes (SMBH)” given “broad hydrogen lines (FWHM $\gtrsim 1000\,\mathrm{km\,s^{-1}}$)”, while also differing from typical AGN in several ways.[^5]  
- Dense gas envelopes are proposed to explain strong Balmer breaks and Balmer absorption, motivating sensitive spectroscopy of GN-9771 to probe envelope properties and their impact on BH-mass inference.[^6]  

## Data

- The study uses **JWST/NIRSpec IFU** spectroscopy (Cycle 4, PID 5664), with PRISM ($R\approx 100$) and G395H ($R\approx 3000$) observations of GN-9771.[^7]  
- GN-9771 was observed for 9.5 hours (Jan 20, 2025), with exposure times of 6.5 ks (PRISM) and 18.2 ks (G395H/F290LP).[^8]  
- Reduction employed the STScI JWST pipeline (v1.17.1) with additional steps (e.g. 1/$f$ noise correction, snowblind, background modelling), and the source is “spatially unresolved” with an optimally extracted 1D spectrum.[^9]  

## Method

- The systemic redshift is set by fitting the narrow $[\mathrm{O\,III}]$ doublet (fixed ratio 2.98), yielding $z=5.53453\pm 0.00004$ and $\mathrm{FWHM}=202\pm 5\,\mathrm{km\,s^{-1}}$.[^10]  
- Balmer lines $H\alpha$ and $H\beta$ are fit with a “fiducial model” comprising “a narrow Gaussian emission, a Gaussian absorber, a broad exponential, and a host Gaussian component”.[^11]  
- Optical $[\mathrm{Fe\,II}]$ is modelled using NIST forbidden transitions with multiplet ratios set by $r_n\propto (2J_i+1)\lambda^{-1}A_{ki}$, and interpreted with **Cloudy** photoionisation grids spanning $n_{\mathrm{H}}=10^{6}\text{--}10^{14}\,\mathrm{cm^{-3}}$ and $N_{\mathrm{H}}=10^{21}\text{--}10^{26}\,\mathrm{cm^{-2}}$.[^12]  

## Results

- The Balmer break strength is measured as $f_{\nu,4000\text{--}4100}/f_{\nu,3620\text{--}3720}=2.5\pm 0.04$.[^13]  
- The Balmer profiles show “prominent non-Gaussian wings” and “P Cygni profiles” in the cores; $H\alpha$ wings are “very well described by a simple exponential” with $\mathrm{FWHM}_{\exp,H\alpha}=1549\pm 5\,\mathrm{km\,s^{-1}}$.[^14]  
- A “plethora” of narrow forbidden $[\mathrm{Fe\,II}]$ lines is detected; they have $\mathrm{FWHM}=464\pm 61\,\mathrm{km\,s^{-1}}$ and are near the $[\mathrm{O\,III}]$ velocity (offset $-43\pm 15\,\mathrm{km\,s^{-1}}$).[^15]  

## Discussion

- The paper concludes that strong $[\mathrm{Fe\,II}]$ arises from “a warm ($T_{\mathrm{e}}\approx 7000\,\mathrm{K}$) outer layer” with $n_{\mathrm{H}}\approx 10^{9\text{--}10}\,\mathrm{cm^{-3}}$, $n_{\mathrm{e}}\approx 10^{7\text{--}8.5}\,\mathrm{cm^{-3}}$, and $N_{\mathrm{H}}\approx 10^{24}\,\mathrm{cm^{-2}}$.[^16]  
- The observed Balmer ratios $H\alpha/H\beta=10.4\pm 0.3$ and $H\gamma/H\beta=0.14\pm 0.03$ are far from Case B, and dust attenuation is disfavoured because inferred $E(B-V)$ values are “in strong tension”.[^17]  
- The narrow $[\mathrm{O\,III}]$ width is used to infer a host-galaxy dynamical mass $\log_{10}(M_{\mathrm{dyn}}/M_\odot)=9.6\pm 0.5$, and a narrow $H\gamma$ flux (under stated assumptions) implies $\mathrm{SFR}\sim 5\,M_\odot\,\mathrm{yr^{-1}}$.[^18]  

## Weaknesses

- The authors caution that the **Cloudy** setups are “empirically motivated phenomenogical models whose main goal is to provide qualitative insights”.[^19]  
- UV $\mathrm{Fe\,II}$ template fitting is explicitly poor ($\chi^2_\nu=7.5$), and the PRISM resolution “hinders a assessment” of UV $\mathrm{Fe\,II}$ transitions.[^20]  
- The paper notes that “more tailored radiative transfer simulations” are required to explain detailed Balmer decrements and profile differences, beyond the empirical fitting components used.[^21]  

## Conclusions

- GN-9771 exhibits broad exponential Balmer wings and P Cygni cores, consistent with Thomson scattering and high Balmer opacity in dense gas.[^22]  
- Photoionisation modelling supports a dense, high-column warm layer ($n_{\mathrm{H}}=10^{9\text{--}10}\,\mathrm{cm^{-3}}$, $N_{\mathrm{H}}>10^{24}\,\mathrm{cm^{-2}}$) producing $[\mathrm{Fe\,II}]$ and a Balmer break similar to that observed.[^23]  
- The findings imply Balmer lines are not reliable virial tracers in LRDs, and BH masses inferred from standard virial calibrations may be overestimated.[^24]  

## Future work

- The authors defer exploring variations in the incident spectrum and degeneracies with host-galaxy contributions “to the future when better high-resolution data is available in the UV and blue optical regime”.[^25]  
- They state that “more tailored radiative transfer simulations” are needed to study the Balmer decrement and line-profile differences in detail.[^26]  
- The prevalence and correlations of optical $[\mathrm{Fe\,II}]$ across LRDs should be tested with “very deep grating spectroscopy in the future”.[^27]  

## Glossary

| Term | Definition |
|---|---|
| **LRD** | “little red dots (LRDs; Matthee et al. 2024)” (Sect. 1, p.1)[^28] |
| **Balmer break** | “define it as $f_{\nu,4000-4100}/ f_{\nu,3620-3720}$” (Sect. 3.2, p.5)[^29] |
| **P Cygni profile** | “the core of the lines (the central $\pm 1000\,\mathrm{km\,s^{-1}}$) resemble P Cygni profiles.” (Sect. 3.1, p.3)[^30] |
| **$[\mathrm{Fe\,II}]$** | “unusually narrow [Fe ii] emission lines” (Sect. 3.3, p.6)[^31] |
| **BH\*** | “Black Hole Star models; BH*; Naidu et al. 2025” (Sect. 5, p.10)[^32] |
| **Ionisation parameter** $U$ | “varying the ionization parameter $\log_{10} U$ from $-4$ to $0$” (Sect. 4.1, p.8)[^33] |

## Tags

#JWST #NIRSpec #FRESCO #JADES #RUBIES

#GalaxiesActive #GalaxiesHighRedshift

## References

[^1]: "To dissect the structure of LRDs, we analyzed new deep JWST/NIRSpec PRISM and G395H spectra of FRESCO-GN-9771, one of the most luminous known LRDs at z = 5.5. These spectra reveal a strong Balmer break, broad Balmer lines, and very narrow [O iii] emission." (Abstract, p.1)

[^2]: "We revealed a forest of optical [Fe ii] lines, which we argue are emerging from a dense (nH = 109−10 cm−3) warm layer with electron temperature Te ≈ 7000 K. The broad wings of Hα and Hβ have an exponential profile due to electron scattering in this same layer." (Abstract, p.1)

[^3]: "The high Hα : Hβ : Hγ flux ratio of ≈ 10.4 : 1 : 0.14 is an indicator of collisional excitation and resonant scattering dominating the Balmer line emission." (Abstract, p.1) ; "Our findings indicate that Balmer lines are not directly tracing the gas kinematics near the SMBH and that the BH mass scale is likely much lower than virial indicators suggest." (Abstract, p.1)

[^4]: "JWST has identified a population of faint compact broad Balmer line emitters at redshifts z = 2–9" (Sect. 1, p.1) ; "that are often due to a Balmer break" (Sect. 1, p.1) ; "hence the name little red dots (LRDs; Matthee et al. 2024)." (Sect. 1, p.1)

[^5]: "The mechanism that powers the emission of LRDs has been the subject of significant debate" (Sect. 1, p.2) ; "but is primarily thought to be driven by accretion onto supermassive black holes (SMBH) due to their compactness and their broad hydrogen lines (FWHM ≳ 1000 km s−1)" (Sect. 1, p.2) ; "However, the spectra of LRDs display several differences with quasars and other types of AGN." (Sect. 1, p.2)

[^6]: "The presence of a very dense gas (nH ≳ 108 cm−3) appears to play an important role in the spectra of LRDs." (Sect. 1, p.2) ; "Although dense gaseous envelopes appear to be an important constituent of LRDs, many questions remain regarding the properties of the envelopes" (Sect. 1, p.2) ; "One of the key goals of our program is to obtain sensitive spectroscopy at the highest resolution possible for JWST covering both the Hα and the Hβ lines." (Sect. 1, p.2)

[^7]: "We used JWST/NIRSpec IFU spectroscopy from the Cycle 4 program PID 5664 (PI Matthee)." (Sect. 2, p.3) ; "The IFU mode was used with the PRISM (R ≈ 100) and high-resolution grating (R ≈ 3000) G395H dispersers." (Sect. 2, p.3)

[^8]: "GN-9771 was observed for 9.5 hours on Jan 20, 2025." (Sect. 2, p.3) ; "The PRISM observations were used to characterize the rest-frame UV-to-optical spectrum and total 6.5 ks of exposure time." (Sect. 2, p.3) ; "The G395H/F290LP grating observations were designed to obtain sensitive high-resolution spectra for the Hβ, [O iii] and Hα emission lines and total 18.2 ks of exposure time." (Sect. 2, p.3)

[^9]: "The data reduction was completed using the STScI JWST pipeline version 1.17.1 with Calibration Reference Data System version 12.0.9" (Sect. 2, p.3) ; "We find that GN-9771 appears spatially unresolved across the full wavelength coverage of our PRISM data" (Sect. 2, p.3) ; "Our measurements are therefore based on a 1D spectrum that is optimally extracted" (Sect. 2, p.3)

[^10]: "We thus obtain a redshift of z = 5.53453 ± 0.00004, and a Gaussian width of FWHM 202 ± 5 km s−1." (Sect. 3.1, p.3) ; "We fit two Gaussians with a fixed ratio of 2.98" (Sect. 3.1, p.3)

[^11]: "We fit the Hα and Hβ lines of GN-9771 to a model consisting of a narrow Gaussian emission, a Gaussian absorber, a broad exponential, and a host Gaussian component. We call this our fiducial model." (Sect. 3.1, p.3)

[^12]: "Within a forbidden transition multiplet, the relative strength of the component n can be obtained as rn ∝ (2Ji + 1) λ−1 Aki , (1)" (Sect. 3.3, p.6) ; "We use Cloudy (version 23.01, last described in Chatzikos et al. 2023; Gunasekera et al. 2023)" (Sect. 4.1, p.8) ; "We simulate clouds with plane-parallel geometry, in a fairly broad range of hydrogen density nH = 106 to 1014 cm−3 and hydrogen column density NH = 1021 to 1026 cm−2" (Sect. 4.1, p.9)

[^13]: "For GN-9771, the Balmer break strength is 2.5 ± 0.04" (Sect. 3.2, p.6) ; "define it as fν,4000−4100/ fν,3620−3720" (Sect. 3.2, p.5)

[^14]: "The emission lines have prominent non-Gaussian wings" (Sect. 3.1, p.3) ; "whereas the core of the lines (the central ±1000 km s−1) resemble P Cygni profiles." (Sect. 3.1, p.3) ; "The broad wings of the Hα line are very well described by a simple exponential" (Sect. 3.1, p.4) ; "The best-fit exponential has a width of FWHMexp,Hα = 1549 ± 5 km s−1." (Sect. 3.1, p.4)

[^15]: "one can also notice a plethora of other fainter lines, most of which are emission lines from transitions from the singly ionized iron atom Fe ii." (Sect. 3.3, p.6) ; "We find that the [Fe ii] lines are at a similar velocity as the [O iii] λλ4960, 5008 velocity (offset by −43 ± 15 km s−1)." (Sect. 3.3, p.7) ; "The [Fe ii] lines have an intermediate width of FWHM = 464 ± 61 km s−1" (Sect. 3.3, p.7)

[^16]: "We conclude that the strong [Fe ii] emission lines of GN-9771 arise from a warm (Te ≈ 7000 K) outer layer of dense (nH ≈ 109–10cm−3) and partially ionized (ne ≈ 107–8.5 cm−3) gas with high column density (NH ≈ 1024 cm−2)." (Sect. 5.1, p.11)

[^17]: "The ratio of the observed Balmer lines Hα/Hβ = 10.4±0.3" (Sect. 5.2, p.11) ; "and Hγ/Hβ = 0.14 ± 0.03" (Sect. 5.2, p.11) ; "Moreover, the attenuation coefficients obtained from both Balmer ratios are in strong tension with each other, making it unlikely that dust attenuation is driving the observed Balmer decrements." (Sect. 5.2, p.11)

[^18]: "The [O iii] width implies a dynamical mass of log10(Mdyn/M⊙) = 9.6 ± 0.5" (Sect. 6.1, p.13) ; "Assuming Case B ratios, and disregarding dust attenuation in the host galaxy, this flux would correspond to a star formation rate of ∼ 5 M⊙ yr−1" (Sect. 6.1, p.13)

[^19]: "We caution that these models are empirically motivated phenomenogical models whose main goal is to provide qualitative insights into the conditions and structure of the gas around LRDs." (Sect. 4, p.8)

[^20]: "However, the low resolution (R ∼ 100) of the PRISM spectrum hinders a assessment of the multiple Fe ii transitions." (Sect. 3.4, p.8) ; "The template fitting does not provide a satisfactory result (χ2ν = 7.5; Fig. 6)" (Sect. 3.4, p.8)

[^21]: "Generally, more tailored radiative transfer simulations (e.g., building upon the model presented in Chang et al. 2025), incorporating the effects of collisional excitation, are required to further study the exact conditions yielding the high Hα/Hβ ratios and explaining the detailed differences in the line-profiles." (Sect. 5.2, p.12)

[^22]: "GN-9771’s Hα and Hβ emission line profiles are well described by a model consisting of broad exponential wings up to at least ± ∼ 7000 km s−1, and a P Cygni profile in the line cores (± ∼ 1000 km s−1). The exponential wings are in line with broadening by Thomson scattering, while the P Cygni cores may originate in a dense gas layer with high opacity to Balmer transitions." (Sect. 7, p.16)

[^23]: "A BH*-like setup yields a best fit to the [Fe ii] ratios for a gas density of nH = 109–10 cm−3 and column densities NH > 1024 cm−2, while still producing a Balmer break with a strength that is similar to what is observed." (Sect. 7, p.16)

[^24]: "Virial BH mass scaling relations not applicable." (Table 4, p.14) ; "Our findings indicate that Balmer lines are not directly tracing the gas kinematics near the SMBH and that the BH mass scale is likely much lower than virial indicators suggest." (Abstract, p.1)

[^25]: "we defer the detailed investigation where the incident spectrum is also varied to the future when better high-resolution data is available in the UV and blue optical regime." (Sect. 4.1, p.9)

[^26]: "Generally, more tailored radiative transfer simulations (e.g., building upon the model presented in Chang et al. 2025), incorporating the effects of collisional excitation, are required" (Sect. 5.2, p.12)

[^27]: "The variation among the strength of the optical [Fe ii] emission, and the relation with other LRD properties (e.g., strength of the Balmer break, emission line properties etc.) should be tested with very deep grating spectroscopy in the future" (Sect. 6.3, p.16)

[^28]: "hence the name little red dots (LRDs; Matthee et al. 2024)." (Sect. 1, p.1)

[^29]: "define it as fν,4000−4100/ fν,3620−3720," (Sect. 3.2, p.5)

[^30]: "whereas the core of the lines (the central ±1000 km s−1) resemble P Cygni profiles." (Sect. 3.1, p.3)

[^31]: "unusually narrow [Fe ii] emission lines." (Sect. 3.3, p.6)

[^32]: "photoionized slab models (from now on, also referred to as Black Hole Star models; BH*; Naidu et al. 2025)" (Sect. 5, p.10)

[^33]: "We set the normalization of the incident SED varying the ionization parameter log10 U from −4 to 0 with 1 dex steps." (Sect. 4.1, p.8)