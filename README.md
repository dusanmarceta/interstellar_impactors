# Interstellar impactors

This project focuses on modeling and analyzing the expected orbital elements, radiants, and velocities of interstellar objects impacting Earth.
The results are published in **Seligman, Darryl Z.; Marčeta, Dušan; Peña-Asensio, Eloy: The Distribution of Earth-Impacting Interstellar Objects, The Astronomical Journal, 2025**.

This repository contains the simulation files and the resulting datasets used to generate the figures presented in the paper.
## Content

### Source files

- interstellar_impactors.py — generates a synthetic population of ISOs and identifies potential Earth impactors.

- synthetic_population.py — contains the algorithm for generating synthetic ISOs according to Marčeta (2023).

- auxiliary_functions.py — auxiliary functions such as conversions between Cartesian state vectors and orbital elements, and vice versa.

- merging_results.py — since interstellar_impactors.py is designed to run in embarrassingly parallel mode, this file merges the output from all parallel runs.

- parallel.sh — executes the defined number of parallel simulations, each on a different core.

### Results folder

- interstellar_imapctors_0.txt — contains the output from one of the individual simulation runs.

- impactors_R=1.csv — merged dataset from all runs used for generating the figures in the paper.


### Output columns

#### The column names and a brief description of each in the file impactors_R=1.csv are given below

- column 0 (q) — perihelion distance of the impactors (au)

- column 1 (ecc) — eccentricity of the impactors

- column 2 (H_init) — initial hyperbolic anomaly of the impactors (rad)

- column 3 (M_impact) — mean anomaly at impact (rad)

- column 4 (moid) — minimum orbit intersection distance with Earth (au)

- column 5 (inc_e) — inclination relative to the ecliptic (rad)

- column 6 (inc_g) — inclination relative to the Galactic plane (rad)

- column 7 (Omega) — longitude of ascending node (rad)

- column 8 (omega) — argument of perihelion (rad)

- column 9 (v_inf) — velocity at infinity (m/s)

- column 10 (v_hc) — heliocentric impact velocity (m/s)

- column 11 (v_gc) — geocentric impact velocity (m/s)

- column 12 (R_eff) — inflated effective radius of the Earth (au)

- column 13 (RA_radiant) — right ascension of the radiant (rad)

- column 14 (DEC_radiant) — declination of the radiant (rad)

- column 15 (sun_longitud) — solar longitude at impact (rad)

### Generation of Figures

#### For each figure, the column name and its index (starting from zero) used for that figure are provided below

- Fig. 1: radiant_RA (13), radiant_DEC (14)

- Fig. 2: radiant_RA (13), radiant_DEC (14), color = v_imp_gc (11)

- Fig. 3, upper panel: v_inf_imp (9)

- Fig. 3, lower panel: v_imp_gc (11), v_hc_1au (10)

- Fig. 4, upper panel: inclination (wrt ecliptic) (5), inclination (wrt galactic plane) (6)

- Fig. 4, middle panel: perihelion distance (0)

- Fig. 4, lower panel: eccentricity (1)

- Fig. 5, upper panel: argument of perihelion (8)

- Fig. 5, middle panel: longitude of ascending node (7)

- Fig. 5, lower panel: hyperbolic anomaly (2)

- Fig. 6: v_hc_1au (9), eccentricity (1)

- Fig. 7: inclination (wrt ecliptic) (5), perihelion distance (0)

- Fig. 9, upper panel: sun_longitude (15), radiant_RA (13), color = v_imp_gc (10)

- Fig. 9, lower panel: sun_longitude (15), radiant_DEC (14) (lower), color = v_imp_gc (10)

- Fig. 10: sun_longitude (15)

 
All histograms should be weighted by v_gc * residence_time (columns 11 and 16).


## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/dusanmarceta/Interstellar_impactors.git
   ```

2. Edit the first three lines in parallel.sh according to your requirements.

3. Run the script to execute the simulations.

# License

This repository is fully open and free to use.

- **Software** (Python scripts, code) is released under the **MIT License**.
- **Data** (CSV files, simulation outputs) is released under the **CC0 1.0 Universal (Public Domain Dedication)**.

You are free to use, copy, modify, and distribute both code and data for any purpose without restriction. 

**Please cite the following reference when using the data or code in publications or projects:**

Seligman, Darryl Z.; Marčeta, Dušan; Peña-Asensio, Eloy — *The Distribution of Earth-Impacting Interstellar Objects*.

# Contributors
- Dusan Marceta, Department of Astronomy, Faculty of Mathematics, University of Belgrade, Serbia
- Darryl Z. Seligman, NSF Astronomy and Astrophysics Postdoctoral Fellow; Department of Physics and Astronomy, Michigan State University, USA
- Eloy Peña-Asensio, Department of Applied Mathematics and Aerospace Engineering, Universitat d'Alacant, Spain; Department of Aerospace Science and Technology, Politecnico di Milano, Italy

# Contact

For questions or collaboration, contact
Dusan Marceta (Department of Astronomy, University of Belgrade), dusan.marceta@matf.bg.ac.rs

