import numpy as np
from scipy.optimize import brentq, newton
import matplotlib.pyplot as plt

__all__ = [ 'getCurve' ]

# Function to find root of EOS to produce the curve chCond vs Beta
def scalarfun( yData, omega, xData, mData  ):
    '''
        @arg yData (array)  : Array containing the experimental data corresponding.
                              In our case, it corresponds to the quantiy defined as
                              y = barPsiPsi
        @arg omega (array)  : Array containing the fitted parameters used to define
                              the function.
        @arg xData (array)  : Array containing the data corresponding to the x axis
                              in the calculation. In our case it corresponds to the
                              quantity x = \beta.
        @arg mData (array)  : Array containing the data corresponding to the parameter
                              m.

        return              : Array containing the Equation of State evaluated at
                              each value of xData, yData, mData for a given omega.
    '''

    return omega[0] * ( xData - omega[1] ) * yData ** omega[2] + \
           omega[3] * yData ** omega[4] - mData

def solveEOS( omegaNew, omegaOld, xPoint ):
    '''
        @arg omegaNew (array) : Array containing the new approximation to the fitted
                                parameters obtained using omega +- random( dOmega ).
        @arg omegaOld (array) : Array containing the old values of the fitted
                                parameters, therefore, omega.
        @arg xPoint (float)   : Point in which we want to calculate the function.

        return:               : Value of the implicit function using omegaNew as
                                the parameters evaluated at xPoint. In order to
                                calculate it we find the roots of scalarfun. We
                                use omegaOld to generate a initial approximation
                                to improve convergence. We only calculate it for
                                one mass, as the chiSquare is constant for all the
                                masses given the old parameters are fixed.
    '''

    mLabel = 0.01

    # We just need the first point - ChiSquare is shared
    inGuess = brentq( scalarfun, 0.0, 0.5, \
                      args = ( omegaOld, xPoint, mLabel ) )
    y = newton( scalarfun, inGuess, maxiter = int( 1e5 ), tol = 1e-3,
                args = ( omegaNew, xPoint, mLabel ) )

    return y

if __name__ == '__main__':
    pass
