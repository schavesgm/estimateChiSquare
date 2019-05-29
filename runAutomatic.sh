#!/bin/bash

# Parameters of the calculation

# Value of the third dimension - Domain wall
Lsize=8
# Value of the other two dimensions
nSize=12
# Limits to fit the data
lims=( 0.2 0.6 )

# Run the python script to generate the data
RESULTS=( $( python eos.py $Lsize $nSize ) )

# Get the results automatically
resArr=()
stdArr=()
for (( i = 0; i < 5; i++ ))
do
    resArr+=( $( echo ${RESULTS[$i]} | sed 's/.*=//' ) )
    stdArr+=( $( echo ${RESULTS[( $i + 5 )]} | sed 's/.*=//' ) )
done

# Save data into temporal file as dotfiles
echo ${resArr[@]} > .temp.file
echo ${stdArr[@]} >> .temp.file

# Run the code
sed -i "/lims =/c\    lims = [ ${lims[0]}, ${lims[1]} ]" ./estimateChiSquare.py
python ./estimateChiSquare.py ./savedData_${nSize}_Ls*${Lsize}.dat

