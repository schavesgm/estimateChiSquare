import numpy as np
import matplotlib.pyplot as plt
import fitFunctionNM as fit
from math import isclose
from functools import partial

__all__ = [ 'fitAndPlot', 'estimateNewPoints' ]

def fitAndPlot( f, inParam, xData, yData, lims, labelName, defChi = None ):

    # Fit the data to get the results
    omega, chiSquare = fit.fitOnceNM( f, np.array( inParam ), \
                       xData, yData, getChiSquare = 1, chiSqDef = defChi )

    print( omega, chiSquare )

    # Plot the data
    xPlot = np.linspace( lims[0], lims[1], 100 )

    plt.plot( xData, yData, '.', label = 'Data - %s' %( labelName ) )
    plt.plot( xPlot, f( omega, xPlot ), label = 'Curve - %s' %( labelName ) )


def elimData( oldArr, posElim ):

    newArr = []
    for i in range( oldArr.shape[0] ):
        if not i in posElim:
            newArr.append( oldArr[i] )

    return newArr

def estimateNewPoints( f, omega, dOmega, xData, yData, lims,
                       yError = 0, defChSq = None, moreDim = None ):

    # Store chiSquare values here
    meanChiSquare = []
    stdeChiSquare = []

    # Generate a new point in the xAxis
    xNew = np.linspace( lims[0], lims[1], 100 )

    # Check for equal data - We do not want degeneracy
    elimPos = []
    for i in range( xNew.shape[0] ):
        for j in range( xData.shape[0] ):
            if( isclose( xNew[i], xData[j], abs_tol=0.05 ) ):
                elimPos.append( i )

    # Eliminate the data that is more or less equal
    if elimPos:
        xNew = elimData( xNew, elimPos )

    # Heaviside function
    heav = lambda x: 1 * ( x > 0.5 ) | -1 * ( x <= 0.5 )

    # Define how the function behaves using moreDim and partial
    if moreDim = None:
        getChi = partial( fit.chiSquare, f, omega, xData, yData, yError, defChSq )
    else:
        getChi = partial( fit.chiSquare, f, omega, xData, \
                          yData, yError, defChSq, moreDim )

    for xN in xNew:

        # Generate a new point in the yAxis according to omega
        storeX = []
        for i in range( 30 ):

            # Generate new parameter data according to the errors
            omegaNew = [ omega[i] + \
                         np.random.rand() * dOmega[i] * heav( np.random.rand() ) \
                         for i in range( len( omega ) ) ]

            # Use the parameters to generate new yData
            yN = f( omegaNew, xN )

            # Add this data to xData and yData
            xData = np.append( xData, xN )
            yData = np.append( yData, yN )

            # Calculate the chiSquare with this new point
            storeX.append( fit.chiSquare( f, omega, xData, yData, yError, defChSq ) )

            # Eliminate the last point to keep flowing
            xData = xData[:-1]
            yData = yData[:-1]

        # Calculate the mean values and the standard deviation
        meanChiSquare.append( np.mean( storeX ) )
        stdeChiSquare.append( np.std( storeX ) )


    # Transform into numpy array
    meanChiSquare = np.array( meanChiSquare )
    stdeChiSquare = np.array( stdeChiSquare )

    return meanChiSquare, stdeChiSquare

if __name__ == '__main__':
    pass
