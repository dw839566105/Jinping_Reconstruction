# Jinping reconstruction package

This is Jinping reconstruction package, you need to install *Jinping Simulation and Analysis Package* (JSAP) including its converter first. JSAP code is located at (https://gitlab.airelinux.org/tjjk/jsap). If you do not have the access, please contact orv.tsinghua.edu.cn.

## Code structure
The reconstruction flow is controlled by the `Makefile`.

The code is arange in the following directory

+ Simulation
> Locate in the `Simulation`. Use macro files to simulate the training and validation data.

+ Regression
> Locate in the `Regression`. Based on the training data, obtain the model by different regression method.

+ Reconstruction
> Locate in the `Reconstruction`. Reconstruct the Simulation data. Notice that the raw data has different data structure. The code is at a local repository (https://gitlab.airelinux.org/jinping/production).

+ Draw
> Locate in the `Draw`. View necessary results via pictures and validate different models.

## Examples

To finish the **whole reconstruction** flow, just run

```
make
```

If you want to do it step-by-step, you can begin the **simulation** by

```
make sim
```
or more precisely, you can run `make` + `shell`, `point/x`, `point/y`, `point/z`, `ball` to obtain dataset with different vertex distributions.

Before reconstruction, you need to fit the model, in reconstruction, we use 40\*35 (r, \theta) on PE and 40\*10 (r, \theta) on timing. Obtain the coefficient via

```
make coeff
```

Accrdingly, you can run the reconstruction by 
```
make recon
```
or run `make` + `recon_shell`, `recon_x`, `recon_y`, `recon_z`, `recon_ball` to check different vertex distribution results.

Optionally, you can fit and validate different models. `make coeff_Z` for Zernike, `make coeff_Leg` for the Varying-coefficient method and `make coeff_dLeg` for the double Legendre basis. 
