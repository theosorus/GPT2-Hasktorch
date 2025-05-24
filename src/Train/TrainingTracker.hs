module Train.TrainingTracker where
import Config

data TrainingTracker = TrainingTracker
  { currentEpoch :: Int
  , currentBatch :: Int
  , currentLr :: Double
  , loss :: [Double]
  , accuracy :: [Double]
  , lastModelPath :: String
  , config :: Config
  } deriving (Show, Eq)







