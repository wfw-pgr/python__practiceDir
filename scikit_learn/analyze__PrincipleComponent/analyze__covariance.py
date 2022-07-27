import numpy as np

# ========================================================= #
# ===  analyze__covariance.py                           === #
# ========================================================= #

def analyze__covariance():

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    import sklearn.datasets as ds
    wines    = ds.load_wine()

    # ------------------------------------------------- #
    # --- [2] compare covariance                    --- #
    # ------------------------------------------------- #
    nRows, nColumns = wines.data.shape
    wines.avgs      = np.average( wines.data, axis=0  )
    wines.stds      = np.std    ( wines.data, axis=0  )
    covs1           = np.zeros  ( (nColumns,nColumns) )
    covs2           = np.zeros  ( (nColumns,nColumns) )
    
    for ik in range( nColumns ):
        for jk in range( nColumns ):
            covs1[ik,jk] = np.average( ( wines.data[:,ik] - wines.avgs[ik] )*
                                ( wines.data[:,jk] - wines.avgs[jk] ) )
            covs2[ik,jk] = np.average( ( wines.data[:,ik] * wines.data[:,jk] ) ) - wines.avgs[ik]*wines.avgs[jk]
    covs_ = np.cov( np.transpose( wines.data ) )
    print( covs_.shape, covs1.shape, covs2.shape )
    ik, jk = 1, 1
    print( covs_[ik,jk] )
    print( covs1[ik,jk] * float( nRows ) / float(nRows-1) )
    print( covs2[ik,jk] * float( nRows ) / float(nRows-1) )

    # ------------------------------------------------- #
    # --- [3] display covariance                    --- #
    # ------------------------------------------------- #
    #  -- [3-1] normalization                       --  #
    wines.norm = ( wines.data - np.repeat( wines.avgs[None,:], nRows, axis=0 ) )
    wines.norm =   wines.norm / np.repeat( wines.stds[None,:], nRows, axis=0 )
    wines.covs = np.cov( wines.norm, rowvar=False )
    print( wines.covs )
    print( wines.covs[-1,-1] )

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_axes( [0.16,0.16,0.74, 0.74] )
    ax.imshow( wines.covs[:,:], cmap="jet" )
    for ik in range( nColumns ):
        for jk in range( nColumns ):
            ax.text( ik, jk, "{0:.3f}".format( wines.covs[ik,jk] ) ,\
                     horizontalalignment="center", \
                     verticalalignment  ="center", fontsize=5 )
    fig.show()
    plt.show()

    
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    ret = analyze__covariance()
