# Simulation

You need to install *Jinping Simulation and Analysis Package* (JSAP) including its converter first. JSAP code is located at (https://gitlab.airelinux.org/tjjk/jsap). If you do not have the access, please contact orv.tsinghua.edu.cn.

`DetectorStructure` describe the geometry of 1-ton detector, including the `gdml` files for GEANT4-based simulation.  

`macro` determines the initial particle type, vertex energy and positions.

The basic usage is
```
    JPSim -n OFF -e ON -g $(geometry) -m (macro files) -dn 0 -gpsFixTime ON -o output_file
```

For convinient, we convert `root` to `h5` to save file. The tool is in (https://gitlab.airelinux.org/jinping/root2hdf5). You can also use `PyROOT` or `uproot` to process the data. However, it is not provided yet.

`concat.py` extract the `r, \theta` for later analysis.
