import numpy as np
import matplotlib.pyplot as plt
from math import isclose
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
    xData = np.empty( [dataLoad.shape[0], 2 ] )
    xData[:,0] = dataLoad[:,0]
    xData[:,1] = dataLoad[:,3]
    yData = dataLoad[:,1]
    yErro = dataLoad[:,2]

    # Define the function to fit the data or the chiSquare
    f = lambda: None
    ch = lambda f, omega, x, y, yE: \
                 np.sum( ( omega[0] * ( xData[:,0] - omega[1] ) * \
                 yData ** omega[2] + omega[3] * yData ** omega[4]
                 - xData[:,1] ) / yE ) ** 2 / len( omega )

    inOmega = np.array( [ 2.9449, 0.2580, 1.0226, 6.7638, 3.6188 ] )
    stOmega = np.array( [ 0.4071, 0.0078, 0.0551, 14.9842, 1.4461 ] )

    # Space to explore
    lims = [ 0.25, 1 ]

    meanChi, stdeChi = pfe.estimateNewPoints( f, inOmega, stOmega,
                                              xData, yData,
                                              lims, defChSq = ch )

    print( meanChi, stdeChi )
    # # Piece of code to generate fake data

    # f = lambda omega, x: omega[0] * x + omega[1] * x ** 2 + \
    #                      omega[0] * omega[1] * np.cos( x + omega[1] )

    # # ch = lambda f, args, xData, yData, yError: 1

    # # Omega - TRUE PARAMETERS
    # omega = [ 1, np.pi / 4 ]
    # dOmega = [ 0.05, 0.05 ]
    # inParam = [ 0.95, np.pi / 4.4 ]

    # # Generate the data
    # xData, yData = [], []
    # for i in range( numPoints ):
    #     xData.append( np.random.rand( ) * 2 * np.pi )
    #     yData.append( f( omega, xData[i] ) + np.random.normal( 0, 0.01 ) )

    # xData = np.array( xData )
    # yData = np.array( yData )

    # # Limits of the data
    # lims = [ 0, 2 * np.pi ]

    # meanChi, stdeChi = pfe.estimateNewPoints( f, omega, dOmega, xData, yData, lims  )
    # xPlot = np.linspace( lims[0], lims[1], meanChi.shape[0] )
    # plt.errorbar( xPlot, meanChi, yerr = stdeChi )
    # plt.show()

    # # Old plots
    # # pfe.fitAndPlot( f, inParam, xData, yData, lims, 'Old', ch )
    # pfe.fitAndPlot( f, inParam, xData, yData, lims, 'Old' )

    # # Add the new point
    # # minX = ( meanChi - stdeChi ).argmin()
    # minX = meanChi.argmax()
    # xData = np.append( xData, xPlot[minX] )
    # yData = np.append( yData, f( omega, xPlot[minX] + np.random.normal( 0, 0.1 ) ) )

    # # New plots
    # # pfe.fitAndPlot( f, inParam, xData, yData, lims, 'New', ch )
    # pfe.fitAndPlot( f, inParam, xData, yData, lims, 'New' )

    # plt.plot( xPlot, f( omega, xPlot ), label = '%s' %( 'Curve - True' ) )
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
