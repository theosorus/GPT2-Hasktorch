{-# LANGUAGE DeriveGeneric #-}
module Train.TrainingTracker where
import Config
import GHC.Generics         (Generic)
import Data.Aeson          (ToJSON, FromJSON, encode, decode)
import qualified Data.ByteString.Lazy as BL

import Model.GPT

data TrainingTracker = TrainingTracker
  { currentEpoch :: Int
  , currentBatch :: Int
  , trainLoss :: [Float]
  , validLoss :: [Float]
  , trainAccuracy :: [Float]
  , validAccuracy :: [Float]
  , lastModelPath :: String
  , modelConf :: ModelConfig
  } deriving (Eq,Generic)


initialTrainingTracker :: ModelConfig -> TrainingTracker
initialTrainingTracker config = TrainingTracker
  { currentEpoch = 0
  , currentBatch = 0
  , trainLoss = []
  , validLoss = []
  , trainAccuracy = []
  , validAccuracy = []
  , lastModelPath = ""
  , modelConf = config
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


-- loadModelFromTracker :: FilePath -> IO (Maybe TrainingTracker)
-- loadModelFromTracker path = do
--   tracker <- loadTrainingTracker path
--   case tracker of
--     Just t  -> do
--       let modelPath = lastModelPath t
--     Nothing -> do
--       putStrLn $ "Failed to load training tracker from " ++ path
--       return Nothing





