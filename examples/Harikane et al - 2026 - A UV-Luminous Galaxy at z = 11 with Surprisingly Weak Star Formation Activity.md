# A UV-Luminous Galaxy at z = 11 with Surprisingly Weak Star Formation Activity

Authors: Harikane Y., Pérez-González P. G., Álvarez-Márquez J., Ouchi M., Nakazato Y., Ono Y., Nakajima K., Umeda H., Isobe Y., Xu Y., Zhang Y.

Published: January 2026 ([Link](https://arxiv.org/abs/2601.21833))

## Key Ideas

- JWST/MIRI observations of CEERS2-588 at $z_\mathrm{spec} = 11.04$ reveal a prominent **Balmer break** — the first such detection at $z > 10$ — indicating an evolved stellar population rather than ongoing starburst activity.[^1]
- Deep MIRI/MRS spectroscopy yields non-detections of H$\alpha$ and [O\,iii]$\lambda$5007, with rest-frame equivalent widths constrained to $\lesssim 200$–$400$ Å, implying recent quenching within the last $\sim 10$ Myr.[^2]
- CEERS2-588 is the most massive spectroscopically confirmed galaxy at $z > 10$ without AGN evidence, with $\log(M_*/M_\odot) = 9.1^{+0.1}_{-0.1}$.[^3]
- The inferred gas-phase metallicity is near solar ($12 + \log(\mathrm{O/H}) \simeq 8.6$), exceeding the predictions of all current theoretical models at this redshift.[^4]
- CEERS2-588 represents the first identification of a **post-starburst** (or "mini-quenching") galaxy at $z > 10$, providing direct empirical evidence for bursty, highly efficient star formation in the early Universe.[^5]

## Introduction

- JWST has revealed an unexpectedly large population of luminous galaxies at $z > 10$, challenging pre-launch theoretical models of early galaxy formation.[^6]
- Progress in characterising these systems has been limited by the absence of rest-frame optical diagnostics; CEERS2-588 was selected as a prime target to address this gap given its UV luminosity ($M_\mathrm{UV} = -20.4$ mag) and confirmed spectroscopic redshift of $z_\mathrm{spec} = 11.04$, placing it just $\sim 400$ Myr after the Big Bang.[^7]
- The extended morphology ($r_e \sim 450$ pc) and absence of high-ionisation emission lines in existing NIRSpec data rule out strong AGN activity, classifying CEERS2-588 as a typical extended system.[^8]

## Data

- CEERS2-588 was observed in JWST Cycle 3 (programme #4586) with MIRI imaging in the F560W and F770W filters (total exposure times of 2.5 and 2.8 hr respectively) and with MIRI/MRS spectroscopy targeting [O\,iii]$\lambda$5007 and H$\alpha$ (9.5 and 17.8 hr respectively).[^9]
- NIRCam photometry and NIRSpec PRISM spectroscopy were drawn from the CEERS programme (#1345) and programme #2750, with data products retrieved from the Dawn JWST Archive (DJA).[^10]
- The full photometric dataset spans HST ACS/WFC3 through JWST MIRI F1000W, with the galaxy robustly detected at $\sim 6\sigma$ in both MIRI F560W and F770W.[^11]

## Method

- MIRI/MRS data were reduced using version 1.20.0 of the JWST calibration pipeline (CRDS context 1462), with customised steps for background subtraction; spectra were extracted within a $0\farcs6$-diameter circular aperture with point-spread-function aperture corrections applied.[^12]
- MIRI imaging was reduced using the **Rainbow pipeline** (based on pipeline version 1.19.41, CRDS context 1413), employing a "superbackground" strategy using temporally matched observations to achieve up to 0.8 mag improvement in depth over standard archive products; photometry was performed with SExtractor (version 2.25.3).[^13]
- **SED fitting** was performed with **Bagpipes** (version 1.3.2) using a five-bin non-parametric star formation history, the Kroupa IMF, Calzetti dust attenuation law, and the Inoue et al. IGM attenuation model, with sampling via the **nautilus** nested sampler; the redshift was fixed to $z_\mathrm{spec} = 11.04$.[^14]
- Gas-phase metallicity was estimated via strong-line diagnostics (R2, O32, R23) calibrated against empirical relations, using the detected [O\,ii]$\lambda$3727 flux and an upper limit on H$\beta$ inferred from H$\alpha$ assuming Case B recombination.[^15]

## Results

- Neither H$\alpha$ nor [O\,iii]$\lambda$5007 is detected in the MIRI/MRS spectra, with $3\sigma$ upper limits of $< 2.4 \times 10^{-18}$ and $< 5.1 \times 10^{-19}$ erg s$^{-1}$ cm$^{-2}$ respectively; the MIRI broadband fluxes are therefore dominated by rest-frame optical stellar continuum, not emission lines.[^16]
- The SED exhibits a clear Balmer break (MIRI bands $\sim 0.3$ mag brighter than shorter-wavelength NIRCam measurements), with SED fitting indicating star formation commenced at $z \sim 15$–$25$ followed by rapid quenching within the most recent $\sim 10$ Myr.[^17]
- The H$\alpha$-based star formation rate is $< 4.4\,M_\odot\,\mathrm{yr}^{-1}$, whilst the UV-based rate is $8.2\,M_\odot\,\mathrm{yr}^{-1}$; the ratio $\mathrm{SFR_{H\alpha}/SFR_{UV}} < 0.54$ is significantly lower than other $z > 10$ galaxies, indicative of a recent quenching event.[^18]
- The stellar mass surface density of $\sim 1700\,M_\odot\,\mathrm{pc}^{-2}$ is comparable to local globular clusters and elliptical galaxies, and the integrated star formation efficiency of $\sim 10\%$ exceeds pre-JWST theoretical predictions of $\sim 3$–$5\%$ for halos of equivalent mass.[^19]

## Discussion

- The combination of strong UV continuum and weak H$\alpha$ emission is naturally explained if CEERS2-588 is observed during a post-burst lull, consistent with cosmological simulations that predict recurrent intense bursts interspersed with mini-quiescent phases lasting $\sim 10$–$100$ Myr at $5 < z < 15$.[^20]
- Current theoretical models cannot simultaneously reproduce the extreme stellar mass and the strongly suppressed H$\alpha$ emission, suggesting that quenching mechanisms beyond stellar feedback — such as AGN feedback and/or radiation-driven outflows — may play a role in massive galaxy formation at $z > 10$.[^21]
- The predicted star formation rates of CEERS2-588 at $z \sim 12$–$14$ are comparable to those of GHZ2, PAN-z14-1, and MoM-z14, suggesting these objects may be progenitors of CEERS2-588.[^22]

## Weaknesses

- The non-detection of H$\alpha$ and [O\,iii] relies on the assumption of no instrumental or pointing issues; while the authors verify spatial alignment via a nearby source in the MRS field, the depth of the MIRI/MRS observations may still be insufficient to detect weak but non-zero line flux.[^23]
- Gas-phase metallicity is constrained primarily from [O\,ii]$\lambda$3727 and an inferred upper limit on H$\beta$; the absence of direct H$\beta$ and [O\,iii] detections limits the precision of the metallicity estimate and introduces dependence on assumed Case B recombination.[^24]
- The comparison with theoretical models is complicated by the fact that different simulations are evaluated at slightly different redshifts ($z = 9.84$–$11$), which may introduce systematic offsets in the predicted mass-metallicity and star formation rate relations.[^25]

## Conclusions

- CEERS2-588 is the first galaxy confirmed at $z > 10$ to exhibit a Balmer break, the most massive such galaxy without AGN evidence ($\log(M_*/M_\odot) \simeq 9.1$), and the most metal-rich, with a metallicity $\sim 5$–$10$ times higher than other $z > 10$ galaxies with metallicity measurements.[^26]
- The galaxy represents the first identification of a post-starburst (mini-quenching) phase at $z > 10$, demonstrating that massive galaxies in the first few hundred million years underwent star formation that was both more efficient and more rapidly quenched than predicted by current models.[^27]

## Future Work

- Deeper spectroscopic observations could yield a direct detection of H$\alpha$ or place tighter constraints on the equivalent width, better characterising the timescale and mechanism of quenching.[^28]
- The inferred star formation history extending to $z > 15$ motivates searches for progenitor galaxies at even earlier epochs, consistent with recent reports of galaxy candidates at $z > 15$.[^29]

## Tags

#JWST #MIRI #NIRCam #NIRSpec #CEERS #Bagpipes #SExtractor #Rainbow #SDSS #FirstLight #BlueTides #FIRE-2 #Santa-Cruz-SAM #UniverseMachine #FLARES #Astraeus #Millennium-TNG #GAEA #nautilus #DawnJWSTArchive
#GalaxiesHighRedshift #GalaxiesEvolution #GalaxiesFormation #GalaxiesStarFormation #GalaxiesFundamentalParameters #GalaxiesAbundances #GalaxiesISM #CosmologyEarlyUniverse #CosmologyDarkAgesReionizationFirstStars #InfraredGalaxies

## Glossary

| Term | Definition |
|---|---|
| Balmer break | A discontinuity in a galaxy's spectrum at rest-frame $\sim 3646$ Å caused by hydrogen Balmer series absorption, indicating the presence of an evolved (A-type or older) stellar population |
| Balmer series | Series of hydrogen spectral transitions to/from the $n=2$ energy level; lines include H$\alpha$ ($\lambda$6563 Å), H$\beta$ ($\lambda$4861 Å), H$\gamma$, H$\delta$ |
| Post-starburst / mini-quenching | A phase in which a galaxy has recently undergone rapid cessation of star formation following an intense burst, characterised by strong Balmer absorption but weak nebular emission |
| SED fitting | Spectral energy distribution fitting; the process of comparing observed multi-wavelength photometry and/or spectroscopy to model templates to infer physical galaxy properties |
| Bagpipes | Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation; a Python SED fitting code |
| Rest-frame equivalent width (EW) | A measure of the strength of a spectral line relative to the adjacent continuum, expressed in Å |
| Ionising photon production efficiency ($\xi_\mathrm{ion}$) | The rate of hydrogen-ionising photon production per unit UV luminosity (units: Hz erg$^{-1}$); a measure of how efficiently a galaxy ionises its surrounding gas |
| Star formation main sequence | The empirical correlation between star formation rate and stellar mass observed in star-forming galaxies across cosmic time |
| Gas-phase metallicity | The abundance of heavy elements (typically measured as oxygen abundance, $12 + \log(\mathrm{O/H})$) in the ionised interstellar gas of a galaxy |
| Strong-line diagnostics | Empirical calibrations relating ratios of bright nebular emission lines (e.g. R2, O32, R23) to gas-phase metallicity |
| R2 | The strong-line ratio [O\,ii]$\lambda$3727/H$\beta$, used as a metallicity diagnostic |
| O32 | The strong-line ratio [O\,iii]$\lambda$5007/[O\,ii]$\lambda$3727, sensitive to ionisation parameter and metallicity |
| R23 | The strong-line ratio $(\mathrm{[O\,iii]}\lambda\lambda4959,5007 + \mathrm{[O\,ii]}\lambda3727)/\mathrm{H}\beta$ |
| Bursty star formation | A mode of star formation characterised by short, intense episodes of rapid stellar mass build-up interspersed with quiescent intervals |
| MIRI/MRS | Mid-Infrared Instrument / Medium Resolution Spectrometer on JWST; an integral field spectrograph covering $\sim 5$–$28$ $\mu$m |
| Lyman break | A sharp drop in the UV spectrum of a galaxy at rest-frame $\lambda < 912$ Å due to absorption of ionising photons by neutral hydrogen |
| Case B recombination | An approximation for hydrogen recombination in nebulae assuming all Lyman series photons are immediately reabsorbed, used to predict line ratios such as H$\alpha$/H$\beta$ |
| Star formation efficiency ($f_\mathrm{SF}$) | The fraction of available baryonic mass in a dark matter halo that has been converted into stars |
| Little red dots | A class of compact, red high-redshift JWST sources tentatively associated with heavily reddened AGN or compact starbursts |
| IMF | Initial mass function; the distribution of stellar masses at formation. Kroupa and Chabrier IMFs are two common parameterisations |

## References

[^1]: "Both the NIRCam and MIRI photometry and NIRSpec spectroscopy indicates the existence of a prominent Balmer break. This represents the first clear detection of a Balmer break in a galaxy at z > 10." (Fig. 2 caption, p.4)

[^2]: "Neither [O iii]λ5007 nor Hα is significantly detected in the MIRI/MRS spectra, with 3σ upper limits of < 2.4×10−18 and < 5.1×10−19 erg s−1cm−2, respectively. The inferred rest-frame equivalent widths are constrained to be ≲ 200 − 400 Å, substantially lower than those measured for comparably luminous galaxies at high redshift." (p.3)

[^3]: "The measured stellar mass, log(M∗/M⊙) ≃ 9.1, makes CEERS2-588 the most massive galaxy known at z > 10 without evidence for AGN activity." (p.4)

[^4]: "Such a massive and metal-rich system has not previously been reported at z > 10 and is not predicted by current theoretical models." (p.4)

[^5]: "CEERS2-588 therefore represents the first identification of a galaxy in a post-starburst (or 'mini-quenching') phase at z > 10." (p.4)

[^6]: "One of the major discoveries by the James Webb Space Telescope (JWST) is the identification of a large population of luminous galaxies at z > 10, challenging theoretical models for early galaxy formation." (Abstract, p.1)

[^7]: "Here we present deep JWST/MIRI observations of a UV-luminous galaxy at z = 11.04, CEERS2-588, only 400 Myr after the Big Bang." (Abstract, p.1)

[^8]: "Its extended morphology, with an effective radius of re ∼ 450 pc, together with the non-detection of high-ionization emission lines in the NIRSpec spectrum, indicates the absence of strong active galactic nucleus (AGN) activity in CEERS2-588." (p.2)

[^9]: "The total on-source exposure times were 9.5 hours for the Medium (B) grating and 17.8 hours for the Short (A) grating." (Methods, p.7); "MIRI imaging was obtained in the F560W and F770W filters, with total exposure times of 2.5 and 2.8 hours, respectively." (Methods, p.7)

[^10]: "NIRCam and NIRSpec observations of CEERS2-588 were obtained from the CEERS program (#1345; PI: S. Finkelstein) and program #2750 (PI: P. Arrabal Haro), respectively. We used data reduced and publicly released through the DAWN JWST Archive." (Methods, p.8)

[^11]: "CEERS2-588 is detected in both F560W and F770W at ∼ 6σ significance." (Methods, p.8)

[^12]: "The MRS observations were processed using version 1.20.0 of the JWST calibration pipeline and context 1462 of the Calibration Reference Data System (CRDS)." (Methods, p.7)

[^13]: "A key component of the Rainbow workflow is the 'superbackground' strategy, in which a background template is constructed from other images in the same filter taken within a three-month period, and known sources are masked to avoid overestimating the background. This procedure produces a highly uniform background in terms of both level and noise, and has been shown to improve the depth of the final mosaics by up to 0.8 mag compared to standard archive products." (Methods, p.8)

[^14]: "SED fitting was performed using Bagpipes version 1.3.2. Free parameters included stellar mass, star formation history, metallicity, ionization parameter, and dust attenuation, while the redshift was fixed to the spectroscopic value." (Methods, p.9)

[^15]: "From the measured [O ii]λ3727 flux and the upper limit on Hβ inferred from Hα assuming Case B recombination, we obtained a constraint of R2 > 3.5." (Methods, p.9)

[^16]: "Given the low equivalent widths of Hα and [O iii], the MIRI broadband fluxes are not dominated by emission-line contributions. Instead, they trace the rest-frame optical stellar continuum." (p.3)

[^17]: "SED fitting indicates that star formation began at z ∼ 15 − 25, corresponding to ∼ 100 − 300 Myr after the Big Bang, followed by a sharp decline in the star formation rate within the recent ∼ 10 Myr." (p.4)

[^18]: "The ratio of the Hα- to UV-based star formation rate is < 0.54, significantly lower than that observed in other galaxies at z > 10, suggesting a recent quenching event." (Methods, p.9)

[^19]: "The resulting stellar mass surface density of ∼ 1700 M⊙ pc−2 is comparable to that of local globular clusters and elliptical galaxies, systems that are also thought to have formed a large fraction of their stars efficiently at early times." (p.5)

[^20]: "Cosmological simulations at 5 < z < 15 predict star formation histories characterized by recurrent, intense bursts interspersed with mini-quiescent phases lasting ∼ 10−100 Myr, during which galaxies can temporarily exhibit suppressed nebular emission while retaining substantial stellar masses." (p.5)

[^21]: "The inability of current models to simultaneously reproduce both the extreme stellar mass and the strongly suppressed Hα emission suggests that quenching mechanisms beyond stellar feedback may play an important role in massive galaxy formation at z > 10, such as AGN feedback and/or radiation-driven outflows." (p.5)

[^22]: "The predicted star formation rates of CEERS2-588 at z ∼ 12 − 14 are comparable to those of GHZ2, PAN-z14-1, and MoM-z14, suggesting that these galaxies could be progenitors of CEERS2-588." (Methods, p.10)

[^23]: "To verify that the non-detection of the [O iii] and Hα emission lines is not caused by any pointing issues in the MIRI/MRS observations, we collapsed the MIRI/MRS datacubes along the wavelength axis to construct continuum images." (Methods, p.7)

[^24]: "From the measured [O ii]λ3727 flux and the upper limit on Hβ inferred from Hα assuming Case B recombination, we obtained a constraint of R2 > 3.5." (Methods, p.9)

[^25]: "In the right panel of Fig. 3, we show the metallicity predictions from FirstLight (z = 11), FIRE-2 (z = 11), FLARES (z = 10), Astraeus (z = 10), Millennium-TNG (z = 11), and GAEA (z = 9.84)." (Methods, p.10)

[^26]: "CEERS2-588 is the most metal-rich galaxy identified at z > 10, with a metallicity ∼ 5 − 10 times higher than that of other galaxies with metallicity measurements at similar redshifts." (Fig. 3 caption, p.5)

[^27]: "These results reveal that massive galaxies in the first few hundred million years of cosmic history experienced star formation that was both more efficient and more rapidly quenched than predicted by theoretical models." (p.6)

[^28]: "The inferred rest-frame equivalent widths are constrained to be ≲ 200 − 400 Å, substantially lower than those measured for comparably luminous galaxies at high redshift, implying a stellar age of ≳ 20 Myr for an instantaneous burst, or ≳ 100 Myr for a constant star formation history." (p.3)

[^29]: "The inferred star formation history extends to z > 15, implying the presence of galaxies at even earlier epochs, consistent with recent reports of galaxy candidates at z > 15." (Methods, p.10)
