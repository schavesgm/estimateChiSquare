# Estimation of chiSquare

In order to run the code you should use _./runAutomatic.sh_.
This script contains the chain that allows you to generate
the parameters and errors using _eos.py_. Then it runs the
file _estimateChiSquare.py_ to calculate the estimation.

Inside _runAutomatic.sh_ the control parameters are:

Note the condensate data is stored in the following nomenclature:

                    cond_mM_nSize_Lsize

- Lsize: Size of the lattice in the third direction - Domain wall.
- nSize: Size of the other dimension.
- lims : Limits in which you want to explore beta

Sergio Chaves García-Mascaraque.
Mayo de 2019.


