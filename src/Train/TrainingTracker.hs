{-# LANGUAGE DeriveGeneric #-}
module Train.TrainingTracker where
import Config
import GHC.Generics         (Generic)
import Data.Aeson          (ToJSON, FromJSON, encode, decode)
import qualified Data.ByteString.Lazy as BL

data TrainingTracker = TrainingTracker
  { currentEpoch :: Int
  , currentBatch :: Int
  , trainLoss :: [Float]
  , validLoss :: [Float]
  , trainAccuracy :: [Float]
  , validAccuracy :: [Float]
  , lastModelPath :: String
  } deriving (Eq,Generic)


initialTrainingTracker :: TrainingTracker
initialTrainingTracker = TrainingTracker
  { currentEpoch = 0
  , currentBatch = 0
  , trainLoss = []
  , validLoss = []
  , trainAccuracy = []
  , validAccuracy = []
  , lastModelPath = ""
  }

instance Show TrainingTracker where
  show tracker =
    "TrainingTracker { "
    ++ "currentEpoch: " ++ show (currentEpoch tracker) ++ ", "
    ++ "currentBatch: " ++ show (currentBatch tracker) ++ ", "
    ++ "trainLoss: " ++ show (trainLoss tracker) ++ ", "
    ++ "validLoss: " ++ show (validLoss tracker) ++ ", "
    ++ "trainAccuracy: " ++ show (trainAccuracy tracker) ++ ", "
    ++ "validAccuracy: " ++ show (validAccuracy tracker) ++ ", "
    ++ "lastModelPath: " ++ lastModelPath tracker
    ++ " }"

instance ToJSON TrainingTracker
instance FromJSON TrainingTracker

saveTrainingTracker :: FilePath -> TrainingTracker -> IO ()
saveTrainingTracker path tracker =
  BL.writeFile path (encode tracker)

loadTrainingTracker :: FilePath -> IO (Maybe TrainingTracker)
loadTrainingTracker path = do
  content <- BL.readFile path
  return (decode content)





