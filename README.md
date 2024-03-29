# Reweight
Reimplementation of aMD reweighting protocol

## Prerequisites

You need, at a minimum (requirements.txt):

* Python 2.7 (only for the moment)
* NumPy
* H5py
* Matplotlib

## Installation on UNIX

I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed (NumPy, Matplotlib, H5py).

## How-To
1 . First, you need to extract all the dV from the NAMD output.
```bash
python extract_dv.py -d directory_namd_output
```
**Command line options**
* -d/--dir: directory  (or list of directories) with the NAMD output ('\*-prod\*.out')
* -i/--interval: interval we take dV (Default: 1)
* -o/--output: name of the output file (Default: weights.dat)

**Output**
* Weights file (Columns: [timestep dV])

2 . Now, reweighting time ! For that you will need at least a 2D coordinate file (coordinate reactions, obtained by using your favorite reduction method like SPE (right ?))(columns: [frame_idx X Y] or [X Y]) and a weight file with all the dV extracted from NAMD output (or from elsewhere like AMBER) (columns: [timestep dV]). And of course, the number of coordinates (structures) should be equal to the number of dV.
```bash
python reweight.py -c coordinate_2d.txt -w weights.dat -m maclaurin
```
**Command line options**
* -c/--coord: 2D coordinates (columns: [frame_idx X Y] or [X Y])
* -w/--weight: weight file with all the dV (columns: [timestep dV])
* -m/--method: reweighting method (choice: pmf, maclaurin, cumulant) (Default: maclaurin)
* -b/--binsize: size of the histogram's bins (Default: 1)
* --cutoff: remove bins with insufficient number of structures (Default: 0)
* -t/--temperature: temperature (Default: 300)
* --mlorder: Order of the maclaurin serie (Default: 10)
* --emax: energy maximum on the free energy plot (Default: None)

**Outputs**
* Reweight file (Columns: [frame_idx X Y energy])
* Free energy plot
* dV avg, dV std and dV anharmonicity plot
* HDF5 file containing raw data (pmf, histogram, dV avg, dV std, dV anharmonicity)

## Citation
1. https://mccammon.ucsd.edu/computing/amdReweighting/
2. Sinko W, Miao Y, de Oliveira CAF, McCammon JA (2013) Population Based Reweighting of Scaled Molecular Dynamics. The Journal of Physical Chemistry B 117(42):12759-12768.
3. Miao Y, Sinko W, Pierce L, Bucher D, Walker RC, McCammon JA (2014) Improved reweighting of accelerated molecular dynamics simulations for free energy calculation. J Chemical Theory and Computation, 10(7): 2677-2689.

## License
MIT
