import numpy as np
import matplotlib.pyplot as plt
import sys

# My modules
import fitFunctionNM as fit
import performEstimation as pfe

def main():

    # numPoints = 10

    # Load the data from outside
    nameFile = sys.argv[1]

    dataLoad = np.loadtxt( nameFile, skiprows = 1 )

    # Given a 2D function, the data is stored inside the xValues
    xData = dataLoad[:,0]
    yData = dataLoad[:,1]
    yErro = dataLoad[:,2]
    mData = dataLoad[:,3]

    # Define the function to fit the data or the chiSquare
    chiSquare = lambda omega, x, y, yE, mData: \
                       np.sum( ( omega[0] * ( xData[:] - omega[1] ) * \
                       yData ** omega[2] + omega[3] * yData ** omega[4]
                       - mData ) / yE ) ** 2 / len( omega )

    inOmega = np.array( [ 2.9449, 0.2580, 1.0226, 6.7638, 3.6188 ] )
    stOmega = np.array( [ 0.4071, 0.0078, 0.0551, 14.9842, 1.4461 ] )

    # Space to explore
    lims = [ 0.2, 0.6 ]

    meanChi, stdeChi, xNew = pfe.estimateNewPoints( inOmega, stOmega, xData,
                                                    yData, yErro, mData,
                                                    lims, chiSquare )

    [ print( meanChi[i], '+-', stdeChi[i] ) for i in range( len( meanChi ) ) ]

    # Get the maximum value
    argMax = np.argmax( meanChi )

    # Plot the data
    plt.errorbar( xNew, meanChi, yerr = stdeChi, c = 'seagreen' )
    plt.xlabel( r'$\beta$' )
    plt.ylabel( r'$\chi^2 / d.o.f$' )
    plt.fill_between( xNew, meanChi + stdeChi, meanChi - stdeChi, \
                      color = 'mediumaquamarine', alpha = 0.5 )
    plt.axvline( xNew[argMax], color = 'darkred' )
    plt.axhline( meanChi[argMax], color = 'darkred' )
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
