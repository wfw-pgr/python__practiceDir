import numpy            as np


# ========================================================= #
# ===  analyze wine Data                                === #
# ========================================================= #

class analyze__wineData:

    # ------------------------------------------------- #
    # --- Constructor                               --- #
    # ------------------------------------------------- #
    def __init__(self):
        
        # ------------------------------------------------- #
        # --- [1] load datasets                         --- #
        # ------------------------------------------------- #
        import sklearn.datasets as dst
        wines          = dst.load_wine()
        self.wineData  = wines.data
        self.wineRank  = wines.target
        self.labels    = wines.feature_names
        self.rankNames = wines.target_names

        # ------------------------------------------------- #
        # --- [2] call methods                          --- #
        # ------------------------------------------------- #
        # self.showDetails__wineData()
        # self.plot__by1Features()
        self.plot__by2Features()
        
        
    # ========================================================= #
    # ===  Wine Data Analysis                               === #
    # ========================================================= #
    
    def showDetails__wineData( self ):
        
        # ------------------------------------------------- #
        # --- [1] show Data Shape & Label Names         --- #
        # ------------------------------------------------- #
        print()
        print( "[DicisionTree.py] wine Data from scikit learn" )
        print( "[DicisionTree.py] wine Data's shape :: {} ".format( self.wineData.shape ) )
        print( "[DicisionTree.py] wine Rank's shape :: {} ".format( self.wineRank.shape ) )
        print()
        print( "[DicisionTree.py] wine's labels     :: >>> " )
        for ik,label in enumerate( self.labels ):
            print( "[analyze__wineData.py] {0:2}. {1:14} ".format( ik+1, label ) )
        print()
        print( "[DicisionTree.py] wine's ranks      :: >>> " )
        for ik,rank  in enumerate( self.rankNames ):
            print( "[analyze__wineData.py] {0:2}. {1:14} ".format( ik+1, rank ) )
        print()

        # ------------------------------------------------- #
        # --- [2] min., max., avg., std. of the wines   --- #
        # ------------------------------------------------- #
        lformat = " {0:3}. {1:30} ||  {2:>8}   {3:>8}   {4:>8}   {5:>8}"
        dformat = " {0:3}. {1:30} ||  {2:8.4f}   {3:8.4f}   {4:8.4f}   {5:8.4f}"
        print()
        print( "[analyze__wineData.py] each features.... " )
        print()
        print( "---"*28 )
        print( lformat.format( "#.", "feature_names", "min.", "max.", "avg.", "std."  ) )
        print( "---"*28 )

        for ik,label in enumerate( self.labels ):

            dmin, dmax = np.min ( self.wineData[:,ik] ), np.min( self.wineData[:,ik] )
            dave, dstd = np.mean( self.wineData[:,ik] ), np.std( self.wineData[:,ik] )
            print( dformat.format( ik, label, dmin, dmax, dave, dstd ) )
        print( "---"*28 )
        print()
        return()
        

    # ========================================================= #
    # ===  plotting by each feature                         === #
    # ========================================================= #
    def plot__by1Features( self ):
        
        # ------------------------------------------------- #
        # --- [1] plot rank by features                 --- #
        # ------------------------------------------------- #
        import nkUtilities.plot1D         as pl1
        import nkUtilities.load__config   as lcf
        import nkUtilities.configSettings as cfs
        x_,y_                    = 0, 1
        pngFile                  = "png/plot__by1Features_{}.png"
        config                   = lcf.load__config()
        config                   = cfs.configSettings( configType="plot.def", config=config )
        config["plt_xAutoRange"] = True
        config["plt_yAutoRange"] = False
        config["plt_yRange"]     = [ -1.0, 3.0]
        config["plt_linestyle"]  = "none"
        config["plt_markerSize"] = 1.0
        config["plt_marker"]     = "o"
        config["yTitle"]         = "Rank"

        for ik, label in enumerate( self.labels ):
            
            config["xTitle"]     = label
            fig = pl1.plot1D( config=config, pngFile=pngFile.format( label[:5] ) )
            fig.add__scatter( xAxis=self.wineData[:,ik], yAxis=self.wineRank[:], \
                              cAxis=self.wineRank[:] )
            fig.set__axis()
            fig.save__figure()
        
        return()


    # ========================================================= #
    # ===  scatter plot by 2 features                       === #
    # ========================================================= #
    def plot__by2Features( self ):
        
        # ------------------------------------------------- #
        # --- [1] plot rank by features                 --- #
        # ------------------------------------------------- #
        import nkUtilities.plot1D         as pl1
        import nkUtilities.load__config   as lcf
        import nkUtilities.configSettings as cfs
        x_,y_                    = 0, 1
        pngFile                  = "png/plot__by2Features_{0}_{1}.png"
        config                   = lcf.load__config()
        config                   = cfs.configSettings( configType="plot.def", config=config )
        config["plt_xAutoRange"] = True
        config["plt_yAutoRange"] = True
        config["plt_marker"]     = "o"
        
        cmap         = [ "Green", "Blue", "Red" ]
        nLabels      = len( self.labels )
        combinations = [ [int(2*ik),int(2*ik+1)] for ik in range( nLabels//2 ) ]
        if ( nLabels%2 == 1 ):
            combinations += [ [ int(nLabels-1),int(0) ] ]
        
        for ik,combi in enumerate( combinations ):
            label1               = self.labels[combi[0]]
            label2               = self.labels[combi[1]]
            config["xTitle"]     = label1
            config["yTitle"]     = label2
            fig = pl1.plot1D( config=config, pngFile=pngFile.format( label1[:5], label2[:5] ) )
            fig.add__scatter( xAxis=self.wineData[:,combi[0]], yAxis=self.wineData[:,combi[1]], \
                              cAxis=self.wineRank[:], cmap=cmap  )
            fig.set__axis()
            fig.save__figure()
        
        return()

    



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    wine = analyze__wineData()
