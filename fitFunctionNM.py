import numpy as np
import copy as cp
import sys
from functools import partial

__all__ = ['fitOnceNM', 'fitNtimes', 'chiSquare']

def chiSquare( f, args, xData, yData, yError = 0, moreDim = None ):
    '''
        @arg f (function) :  function that we would like to fit with the data.
                             Must return a double
                             and be able to accept numpy arrays as input
        @arg args (array) :  values that we would like to fit
        @arg xData(array) :  xData extracted from the experiment that we would
                             like to fit
        @arg yData(array) :  yData extracted from the experiment that we would
                             like to fit
        @arg yError(array):  error in the measured function used to calculate the
                             fit function
        @arg moreDim(array): array containing more data for N-dimensional functions
                             if it is different from zero, f is expected to accept
                             three arguments.

        return             : double with the chiSquare value at the current status
                             of the coeffs. It retuns the reduced chiSquare value
                             by dividing the chiSquare by the number of degrees of
                             freedom in parameter space
    '''
    if moreDim is None:
        if type( yError ) is int:
            return np.sum( ( f(args, xData) - yData ) ** 2 ) / len(args)
        else:
            return np.sum( ( (f(args, xData) - yData) / yError ) ** 2 ) / len(args)
    else:
        if type( yError ) is int:
            return np.sum( ( f(args, xData, moreDim) - yData ) ** 2 ) / len(args)
        else:
            return np.sum( ( (f(args, xData, moreDim) - yData) / yError ) ** 2 ) \
                   / len(args)



def fitOnceNM( f, initGuess, xData, yData, chiSqDef = None, moreDim = None,
               yError = 0, seed = 1231523, maxIter = 1e6, getChiSquare = 0,
                ):
    '''
        @arg f (function)   : function that we would like to optimize. This function
                              must return a double and be able to get numpy arrays
                              as input.
        @initGuess (array)  : array with the initial guess of the points that
                              minimize the function.
        @xData (array)      : Array containing the data corresponding to the
                              first dimension in the function f(x,y,z...).
        @yData (array)      : Experimental data to fit corresponding to the
                              function f(x,y,z...).

        Optionals:
        @chiSqDef (funciton): Function that defines the chiSquare for your problem.
                              In case it is None, you will use the ones defined in
                              the function chiSquare in this package.getChiSquare
        @moreDim (array)    : More dimension in the function f(x,y,z).
        @yError (array)     : Error associated to the experimental values. If set
                              to zero and chiSqDef is None, we will use the error
                              function.
        @seed (int)         : Seed used in the calculation for random data.
        @maxIter (int)      : Maximum number of iterations calculated in the
                              Nelder-Mead/Amoeba algorithm.
        @getChiSquare (int) : If set to zero, chiSquare is not returned at the final
                              convergence of the algorithm.

        return: tuple     : tuple( Points that minimize the function, minimum
                            value achieved). If getChiSquare is 1, then it returns
                            an array with 2 dimensions, the second one is the
                            chiSquare.
    '''

    # Set the chiSquare definition used
    if chiSqDef is None:
        chiSqDef = chiSquare
    else:
        pass

    if moreDim is None:
        chiCall = partial( chiSqDef, f, xRef, xData, yData, yError )
    else:
        chiCall = partial( chiSqDef, f, xRef, xData, yData, yError, moreDim )

    np.random.seed( seed )

    ## DEFINE THE CONSTANTS OF THE ALGORITHM
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    noImpThreshold = 1E-8
    noImprovementBreak = 1000

    # INIT THE ALGORITHM

    N = initGuess.shape[0]  # Dimension parameter space

    evalFunc = chiSqDef(f,initGuess,xData,yData,yError)
    noImprov = 0
    imValues = [[initGuess, evalFunc]]

    for i in range( N ):
        newPoint = cp.copy(initGuess) + initGuess * np.random.random(N)
        imNewPoint= chiCall()
        imValues.append([newPoint, imNewPoint])

    ### Simplex iteration
    iterStep = 0
    while 1:

        # Order the evaluate function

        ## Sort using the evaluated function as axis
        imValues.sort( key = lambda x: x[1] )
        lowestValue = imValues[0][1]            ## Best value == lowest value

        # Break if mas iteration is achieved
        if maxIter and iterStep >= maxIter:
            bestData = imValues[0][0]           ## Best minimum obtained
            evalChiSQ = imValues[0][1]          ## Value of the chiSquare achieved

            ## Control variable to print out the chiSquare
            if getChiSquare == 0:
                return bestData
            else:
                return bestData, evalChiSQ


        iterStep += 1

        # Break in case of insignificant improvement
        ## print( 'lowest value in iter ', iterStep, 'is ', lowestValue )

        if lowestValue <  evalFunc - noImpThreshold:
            noImprov = 0
            evalFunc = lowestValue
        else:
            noImprov += 1

        if noImprov >= noImprovementBreak:
            bestData = imValues[0][0]
            evalChiSQ = imValues[0][1]

            ## Control variable to print out the chiSquare
            if getChiSquare == 0:
                return bestData
            else:
                return bestData, evalChiSQ

        # Calculate centroid of the points
        centroidPoint = [0.] * N

        for tup in imValues[:-1]:               ## Get all the tuples in the 'matrix'
            for i, c in enumerate(tup[0]):      ## Disect both tuples
                centroidPoint[i] += c / (len(imValues)-1)       ## Sum the centroid

        # REFLECTION
        xRef = centroidPoint + alpha * ( centroidPoint - imValues[-1][0])
        # refEval = chiSqDef(f,xRef,xData,yData,yError)
        refEval = chiCall()

        if imValues[0][1] <= refEval < imValues[-2][1]:

            ## If the lowest value is lower than
            ## the refEval and refEval is lower
            ## than the second largest value

            del imValues[-1]                    ## Remove the last point
            imValues.append([xRef, refEval])    ## Append the reflection as highest
            continue

        # EXPANSION
        if refEval < imValues[0][1]:

            ## If the value of the reflection is
            ## smaller than the lowest value

            xExp = centroidPoint + gamma * ( xRef - imValues[-1][0] )
            # expValue = chiSqDef(f,xExp,xData,yData,yError)
            expValue = chiCall()

            if expValue < refEval:
                ## If the value of the expansion is
                ## less than the reflection value

                del imValues[-1]
                ## We append the expansion to the last
                imValues.append([xExp, expValue])
                continue

            else:
                del imValues[-1]
                ## If not, we append the reflection
                imValues.append([xRef, refEval])
                continue

        # CONTRACTION
        xCont = centroidPoint + rho * ( centroidPoint - imValues[-1][0] )
        # contEval = chiSqDef(f,xCont,xData,yData,yError)
        contEval = chiCall()

        if contEval < imValues[-1][1]:

            ## If the contraction image is smaller
            ## than the highest point, replace it
            del imValues[-1]
            imValues.append([xCont, contEval])
            continue

        # REDUCTION
        nimValues = []

        ## We contract everything using the
        ## lowest value as the fixed point

        for tup in imValues:
            xRed =  imValues[0][0] + sigma * ( tup[0] - imValues[0][0] )
            # redEval = chiSqDef(f,xRed,xData,yData,yError)
            redEval = chiCall()
            nimValues.append( [xRed, redEval] )
        imValues = nimValues

def fitNtimes( f, nReps, initGuess, xData, yData, yError = 0, seed = 1231241 ):
    '''
        Routine that calls "fitOnceNM" nReps times in order to get
        some statistics. The value of initGuess is multiplied by some random value
        in order to randomize the nCalculations and explore the space better.
    '''

    getFit = []
    for i in range( nReps ):
        stepSpace = np.sqrt( np.dot( initGuess, initGuess ) )
        initGuess += stepSpace * np.random.random(len(initGuess))
        getFit.append( fitOnceNM( f, initGuess, xData, yData, yError, seed ) )

    return getFit

if __name__ == "__main__":
    pass
