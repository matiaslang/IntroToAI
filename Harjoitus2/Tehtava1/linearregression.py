# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split

def performance(Y_test, X_test, coeffs, f):
    """
    Tämä funktio arvioi sovitetun käyrän suorituskykyä verrattuna testijoukon datapisteisiin.
    """
    print("The performance of linear fit:")
    print("Mean absolute error={}".format(round(sm.mean_absolute_error(Y_test, f(X_test))), 3))
    print("Mean squared error={}".format(round(sm.mean_squared_error(Y_test, f(X_test))), 3))
    print("Explained variance score={}".format(sm.explained_variance_score(Y_test, f(X_test))))
    print("R2 score={}".format(sm.r2_score(Y_test, f(X_test))))
    y = "Fitted line is form y = "
    for i in range(len(coeffs)):
        if len(coeffs)-i == 2:
            y+= "({})*x + ".format(coeffs[i], (len(coeffs)-i-1))
            continue
        if len(coeffs)-i == 1:
            y+= "({})".format(coeffs[i])
            continue
        y+= "({})*x^{} + ".format(coeffs[i], (len(coeffs)-i-1))
    print(y)

def main():
    inputfile = "data1_sea_level.txt"    # Ladataan datajoukko (vakiona opintopistekertymä datajoukko)
    X = np.loadtxt(inputfile, delimiter=",", usecols=[0])
    Y = np.loadtxt(inputfile, delimiter=",", usecols=[1])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    plt.scatter(X_train, Y_train, color="blue",s=5, marker='o') # Plotataan opetusjoukon näytteet kuvaajaan 
    #-------TÄHÄN SINUN KOODI--------
    coeffs = np.polyfit(X_train, Y_train, 1)
    function = np.poly1d(coeffs)

    x_line = np.linspace(min(X_train), max(X_train))
    y_line = function(x_line)

    plt.scatter(X_test, Y_test, color="red", s=7)
    plt.plot(x_line, y_line, color="green")

    plt.title('Sea level')
    plt.xlabel('Year')
    plt.ylabel('sea level risen from point zero(7000mm)')

    print("Sea level at year 2025 from point zero(7000mm) is: {}mm".format(function(2025)))
    
    #--------------------------------
    performance(Y_test, X_test, coeffs, function)   # Ota kommenttimerkki pois tämän rivin edestä testataksesi luomasi mallin suorituskykyä
    plt.grid()   # Plotataan kuvaajaan ruudukko
    plt.show()   # Näytetään kuvaaja
	
	
if __name__ == '__main__':
    main()
