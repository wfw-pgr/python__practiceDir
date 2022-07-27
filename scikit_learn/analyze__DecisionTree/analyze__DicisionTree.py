import numpy                   as np


# ========================================================= #
# ===  analyze wine data by Decision Tree               === #
# ========================================================= #

def analyze__DecisionTree():

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    import sklearn.datasets as ds
    wines      = ds.load_wine()
    wineData   = wines.data
    wineRank   = wines.target
    itemNames  = wines.feature_names
    rankNames  = wines.target_names

    import sklearn.model_selection as ms
    test_size     = 0.7
    random_state  = 0
    xTrain, xTest = ms.train_test_split( wineData, test_size=test_size, random_state=random_state )
    yTrain, yTest = ms.train_test_split( wineRank, test_size=test_size, random_state=random_state )
    
    # ------------------------------------------------- #
    # --- [2] Define Dicision Tree                  --- #
    # ------------------------------------------------- #
    import sklearn.tree as skt
    DTC     = skt.DecisionTreeClassifier( max_depth=1 )
    DTC.fit( xTrain, yTrain )
    score   = DTC.score( xTest, yTest )
    yPred   =  DTC.predict( xTest )
    correct = np.where( ( yPred == yTest ) )
    wrong   = np.where( ( yPred != yTest ) )
    print( "[analyze__DecisionTree.py] score :: {}".format( score ) )

    # ------------------------------------------------- #
    # --- [3] scatter plot of comparision           --- #
    # ------------------------------------------------- #
    ax1_, ax2_ = 10, 11
    colorNames = [ "lawngreen", "Cyan", "Orange"]
    import matplotlib.colors as mcl
    cmap       = mcl.ListedColormap( colorNames )
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_axes( [0.12,0.12,0.74,0.74] )
    ax.scatter( xTest[:,ax1_]       , xTest[:,ax2_]       , c=yTest   , cmap =cmap   )
    ax.scatter( xTest[:,ax1_][wrong], xTest[:,ax2_][wrong], marker="x", color="black")
    ax.set_xlabel( itemNames[ax1_] )
    ax.set_ylabel( itemNames[ax2_] )
    fig.show()
    plt.show()

    # ------------------------------------------------- #
    # --- [4] export graph tree structure           --- #
    # ------------------------------------------------- #
    import graphviz
    dot_data = skt.export_graphviz( DTC, out_file=None, feature_names=itemNames, \
                                    class_names=rankNames, filled=True, rounded=True)
    graph = graphviz.Source( dot_data )
    graph.render("wine_data", format="png")
    

# ========================================================= #
# ===  experiment__for_max_depth_scan                   === #
# ========================================================= #

def experiment__for_max_depth_scan():

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    import sklearn.datasets as ds
    wines      = ds.load_wine()
    wineData   = wines.data
    wineRank   = wines.target
    itemNames  = wines.feature_names
    rankNames  = wines.target_names

    import sklearn.model_selection as ms
    test_size     = 0.7
    random_state  = 0
    xTrain, xTest = ms.train_test_split( wineData, test_size=test_size, random_state=random_state )
    yTrain, yTest = ms.train_test_split( wineRank, test_size=test_size, random_state=random_state )
    
    # ------------------------------------------------- #
    # --- [2] Define Dicision Tree                  --- #
    # ------------------------------------------------- #
    import sklearn.tree as skt

    max_depth = 20
    scores    = []
    
    for ik in range( max_depth ):
        DTC     = skt.DecisionTreeClassifier( max_depth=ik+1 )
        DTC.fit( xTrain, yTrain )
        score   = DTC.score( xTest, yTest )
        scores.append( score )
        print( "[analyze__DecisionTree.py] score :: {}".format( score ) )

    scores    = np.array( scores )
    iteration = np.arange( max_depth ) + 1.0

    import nkUtilities.plot1D         as pl1
    import nkUtilities.load__config   as lcf
    import nkUtilities.configSettings as cfs
    pngFile                  = "png/maxDepth_scan.png"
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ +0, +max_depth ]
    config["plt_yRange"]     = [ +0, +1.0 ]
    config["xMajor_Nticks"]  = 11
    config["yMajor_Nticks"]  = 11
    fig     = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot( xAxis=iteration[:], yAxis=scores[:], marker="o", linestyle="-" )
    fig.set__axis()
    fig.save__figure()
    


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    analyze__DecisionTree()
    experiment__for_max_depth_scan()
