import os, sys
import numpy as np

# ========================================================= #
# ===  analyze wine data using PCA & DT                 === #
# ========================================================= #

def analyze__PCA_DTC( n_components=13, max_depth=10, test_size=0.6, random_state=1, skip_pca=False ):

    # ------------------------------------------------- #
    # --- [1] load wine data                        --- #
    # ------------------------------------------------- #
    import sklearn.datasets as ds
    wines = ds.load_wine()

    # ------------------------------------------------- #
    # --- [2] precondition :: normalization         --- #
    # ------------------------------------------------- #
    wines.avgs  = np.average( wines.data, axis=0 )
    wines.stds  = np.std    ( wines.data, axis=0, ddof=True )
    avgs_expand = np.repeat ( wines.avgs[None,:], wines.data.shape[0], axis=0  )
    stds_expand = np.repeat ( wines.stds[None,:], wines.data.shape[0], axis=0  )
    wines.norms = ( wines.data - avgs_expand ) / stds_expand

    # ------------------------------------------------- #
    # --- [3] precondition :: PCA                   --- #
    # ------------------------------------------------- #
    if ( skip_pca is False ):
        import sklearn.decomposition as dcm
        wines.pca     = dcm.PCA( n_components=n_components )
        wines.pca.fit( wines.norms )
        wines.reduced = wines.pca.transform( wines.norms )
        print( wines.reduced.shape )
    else:
        wines.reduced = wines.norms
        
    # ------------------------------------------------- #
    # --- [4] Decision Tree Classifier              --- #
    # ------------------------------------------------- #
    #  -- [4-1] split train / test data             --  #
    import sklearn.model_selection as ms
    xTrain, xTest = ms.train_test_split( wines.reduced, test_size=test_size,\
                                         random_state=random_state )
    yTrain, yTest = ms.train_test_split( wines.target , test_size=test_size,\
                                         random_state=random_state )
    print( xTrain.shape, xTest.shape )
    print( yTrain.shape, yTest.shape )
    
    #  -- [4-2] apply DTC                           --  #
    import sklearn.tree as skt
    DTC = skt.DecisionTreeClassifier( max_depth=max_depth, random_state=random_state )
    DTC.fit( xTrain, yTrain )

    # ------------------------------------------------- #
    # --- [5] evalution of DTC                      --- #
    # ------------------------------------------------- #

    accuracy = DTC.score( xTest, yTest )
    
    print( "[analyze__PCA_DTC] [[ Evaluation ]]" )
    print()
    print( "[analyze__PCA_DTC] accuracy      :: {}".format( accuracy ) )
    return( accuracy )
    
    
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] investigate Accuracy                  --- #
    # ------------------------------------------------- #
    nRandom_Try      = 50
    n_components_max = 13
    offset           = 10
    nComponents_Try  = np.arange( 1, n_components_max+1 )
    results          = np.zeros ( ( n_components_max,) )
    
    for ik,nc in enumerate( nComponents_Try ):
        sumvar = 0.0
        for rs in range( offset, offset+nRandom_Try ):
            ret     = analyze__PCA_DTC( n_components=nc, random_state=rs )
            sumvar += float( ret )
        results[ik] = sumvar / float( nRandom_Try )

    results2         = np.zeros ( ( n_components_max,) )
    sumvar = 0.0
    for rs in range( offset, offset+nRandom_Try ):
        ret     = analyze__PCA_DTC( n_components=nc, random_state=rs, skip_pca=True )
        sumvar += float( ret )
    results2[:] = sumvar / float( nRandom_Try )
    
    
        
    # ------------------------------------------------- #
    # --- [2] plot graph                            --- #
    # ------------------------------------------------- #
    import nkUtilities.plot1D         as pl1
    import nkUtilities.load__config   as lcf
    import nkUtilities.configSettings as cfs
    x_,y_                    = 0, 1
    pngFile                  = "png/out.png"
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["plt_xAutoRange"] = True
    config["plt_yAutoRange"] = True
    config["plt_xRange"]     = [ -1.2, +1.2 ]
    config["plt_yRange"]     = [ -1.2, +1.2 ]
    fig     = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot( xAxis=nComponents_Try, yAxis=results , label="w/  PCA" )
    fig.add__plot( xAxis=nComponents_Try, yAxis=results2, label="w/o PCA" )
    fig.set__axis()
    fig.save__figure()

