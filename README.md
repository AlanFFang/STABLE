# STABLE (with BbB Feedback & Detune Frequency Solver Extensions)

This repository is an extended fork of **STABLE** , originally developed by the team at the **National Synchrotron Radiation Laboratory (NSRL)**, University of Science and Technology of China (USTC) from Tianlong He.

---

## Overview of Extensions

To facilitate bunch by bunch feedback for storage rings, this extended version introduces two key modules developed by **Peizhi Fang**:

1. **Bunch-by-Bunch (BbB) Feedback Module**:
   - Models turn-by-turn bunch position/phase pickup.
   - Calculates and applies corrective kicker voltages to effectively damp longitudinal coupled-bunch instabilities.

2. **Detuning Frequency Iteration Solver**:
   - Provides a self-consistent iterative routine to determine optimal harmonic cavity (HC) detuning frequencies and phase settings under uniform fill patterns and beam loading conditions.
---

## Attribution & Original Work

* **Original Developers**: hetianlong-afk, USTC.
* **Original Repository**: https://github.com/hetianlong-afk/STABLE.
* **Original Literature Reference**: He, T. and Z. Bai, Graphics-processing-unit-accelerated simulation for longitudinal beam dynamics of arbitrary bunch trains in electron storage rings. Physical review. Accelerators and beams, 2021. 24(10): p. 104401.
