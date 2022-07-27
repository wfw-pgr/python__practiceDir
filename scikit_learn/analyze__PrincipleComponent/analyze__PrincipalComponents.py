import os, sys
import pandas as pd
import numpy  as np

# ========================================================= #
# ===  analyze__PrincipalComponents                     === #
# ========================================================= #

def analyze__PrincipalComponents():

    # ------------------------------------------------- #
    # --- [1] load wine Data / correlation matrix   --- #
    # ------------------------------------------------- #
    import sklearn.datasets as ds
    wines              = ds.load_wine()
    wines.correlations = np.corrcoef( wines.data, rowvar=False )

    display = False
    if ( display ):
        import nkUtilities.draw__heatmap as hmp
        hmp.draw__heatmap( Data=wines.correlations )
    
    # ------------------------------------------------- #
    # --- [2] apply PCA                             --- #
    # ------------------------------------------------- #
    #  -- [3-1] normalization of data               --  #
    wines.avgs  = np.average( wines.data, axis=0            )
    wines.stds  = np.std    ( wines.data, axis=0, ddof=True )
    avgs_expand = np.repeat ( wines.avgs[None,:], wines.data.shape[0], axis=0 )
    stds_expand = np.repeat ( wines.stds[None,:], wines.data.shape[0], axis=0 )
    wines.norms = ( wines.data - avgs_expand ) / ( stds_expand )
    print( wines.norms.shape )
    
    #  -- [3-2] apply PCA                           --  #
    import sklearn.decomposition as dec
    pca           = dec.PCA( n_components=5 )
    wines.pca     = pca.fit( wines.norms )

    # ------------------------------------------------- #
    # --- [3] show eigen vectors                    --- #
    # ------------------------------------------------- #
    eigenvectors  = wines.pca.components_
    print()
    print( "[pca] [Loadings]       :: " )
    print()
    print( '[pca] pca.components_   is call as "Loadings" ' )
    print( "[pca] Loadings == Mixing Ratio ( or Recipe! ) of the vectors :: ( 13 x nComponents here. )" )
    print( "[pca] components'shape :: {}".format( wines.pca.components_.shape ) )
    print()
    loadings      = pd.DataFrame( wines.pca.components_.T, index=wines.feature_names )
    print( "[pca] Loading          :: " )
    print( loadings )
    print()
    print()

    # ------------------------------------------------- #
    # --- [4] pca transformed ( size reduced ) data --- #
    # ------------------------------------------------- #
    pca_data      = wines.pca.transform( wines.norms )
    print( "[pca] transform wines.data :: " )
    print( "[pca] transformed ( reduced ) data's shape :: {}".format( pca_data.shape ) )
    print()
    print( "[pca] scatter figure   :: " )
    display = True
    if ( display ):
        ik, jk  = 4, 0
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        import matplotlib.colors as mcl
        cmap    = mcl.ListedColormap( ["LawnGreen","RoyalBlue","Orange"] )
        ax.scatter( pca_data[:,ik], pca_data[:,jk], c=wines.target, cmap=cmap )
        fig.savefig( "png/pca_data_scatter_a{0}-a{1}.png".format( ik, jk ) )
    
    return( wines )

    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    Data = analyze__PrincipalComponents()
