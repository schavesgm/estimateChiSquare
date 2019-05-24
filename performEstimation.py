import numpy as np
import matplotlib.pyplot as plt
import fitFunctionNM as fit
from math import isclose
import solveEOS as seos

__all__ = [ 'estimateNewPoints' ]

def elimData( oldArr, posElim ):
    '''
        @arg oldArr (array)  : Array containing all the data -- We want to clean it
        @arg posElim (array) : Array containing the positions of the data we would
                               like to eliminate inside oldArr

        return (array)       : Numpy array containing the values of oldArray minus
                               the points corresponding to the positions included in
                               posElim
    '''

    newArr = []
    for i in range( oldArr.shape[0] ):
        if not i in posElim:
            newArr.append( oldArr[i] )

    return np.array( newArr )

def estimateNewPoints( omega, dOmega, xData, yData, yError, mData, lims,
                       defChSq, numIter = 75, numPoints = 100 ):
    '''
        @arg omega (array) : Array containing the fitted parameters calculated
                             with the N-1 previous points.
        @arg dOmega (array): Array containing the errors in the fitted parameters
                             calculated with N-1 previous points.
        @xData (array)     : Array containing the x data acting in the function
                             to fit as f(x). In our case x = \beta
        @yData (array)     : Array containing the experimental y data used to fit
                             the function via the chiSquare definition. In our
                             case this value is barPsiPsi
        @yError (array)    : Array containing the standard deviation associated
                             with the experimental data y.
        @mData (array)     : Array containing the data corresponding to m in the
                             simulation.
        @lims (array)      : Array with two entries that hold the minimum and
                             maximum values to explore in the x-space. The first
                             one is the minimum, the second one is the maximum.
        @defChSq (function): Function that defines the chiSquare to be minimized
                             in our problem. It must accept the following
                             arguments ( omega, xData, yData, yError, mData ) and
                             return a float.
        Optionals:
        @numIter (int)     : Maximum number of iterations used for each x point.
        @numPoints (int)   : Maximum number of points to be calculated between
                             lim[0] and lim[1]. They are linearly distributed between
                             these points. Note that some of them are removed as
                             degenerate values are not accepted.

        return             : The program returns three different arrays. The first
                             one corresponds to the mean of chiSquares for each
                             position to explore. The second one is the standard
                             deviation associated to this mean value. The third and
                             last is the points corresponding to each chiSquare.
                             For xNew[i], it corresponds meanChiSquare[i] +-
                             stdeChiSquare[i]
    '''

    # Generate a new point in the xAxis
    xNew = np.linspace( lims[0], lims[1], numPoints )

    # Check for equal data - We do not want degeneracy
    elimPos = []
    for i in range( xNew.shape[0] ):
        for j in range( xData.shape[0] ):
            if( isclose( xNew[i], xData[j], abs_tol=0.001 ) ):
                elimPos.append( i )

    # Eliminate the data that is more or less equal
    if elimPos:
        xNew = elimData( xNew, elimPos )

    # Heaviside function
    heav = lambda x: 1 * ( x > 0.5 ) | -1 * ( x <= 0.5 )

    # Generate a new point in the yAxis according to omega
    storeX = np.empty( [numIter] )

    # Define the arrays to hold the data
    meanChiSquare = np.empty( [xNew.shape[0]] )
    stdeChiSquare = np.empty( [xNew.shape[0]] )

    indX = 0
    for xN in xNew:

        for i in range( numIter ):

            # Generate new parameter data according to the errors
            omegaNew = [ omega[i] + \
                         0.25 * np.random.rand() * dOmega[i] * \
                         heav( np.random.rand() ) \
                         for i in range( len( omega ) ) ]

            # Use the parameters to generate new yData -- Need to call brentQ
            yN = seos.solveEOS( omegaNew, omega, xN )

            # Add the point into xData
            xData = np.append( xData, xN )
            yData = np.append( yData, yN )

            # Calculate the chiSquare with this new point
            storeX[i] = defChSq( omegaNew, xData, yData, yError, mData )

            # Eliminate the last point to keep flowing
            yData = yData[:-1]
            xData = xData[:-1]

        meanChiSquare[indX] = np.mean( storeX )
        stdeChiSquare[indX] = np.std( storeX )

        indX += 1

    return np.array( meanChiSquare ) , np.array( stdeChiSquare ), xNew

if __name__ == '__main__':
    pass
