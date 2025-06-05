module TrainingTrackerTest where

import Train.TrainingTracker
import System.Directory (removeFile)

import Model.GPT (ModelConfig(ModelConfig))



import Test.Hspec



trainingTrackerSaveAndLoad :: Spec
trainingTrackerSaveAndLoad = do
    let tt = TrainingTracker { 
        currentEpoch = 10
    , currentBatch = 1024
    , trainLoss = [10.0, 9.5, 8.7]
    , validLoss = [10.0, 9.8, 9.2]
    , trainAccuracy = []
    , validAccuracy = [0.8, 0.85, 0.9]
    , lastModelPath = "Hello"
    , modelConf = ModelConfig 128 2 30000 2 128
  }
    let filePath = "training_tracker_test.json"
        
    it "TrainingTracker must be save and load" $ do
        saveTrainingTracker filePath tt
        loadedTT <- loadTrainingTracker filePath
        removeFile filePath
        case loadedTT of
            Just loaded -> do
                currentEpoch loaded `shouldBe` currentEpoch tt
                currentBatch loaded `shouldBe` currentBatch tt
                trainLoss loaded `shouldBe` trainLoss tt
                validLoss loaded `shouldBe` validLoss tt
                trainAccuracy loaded `shouldBe` trainAccuracy tt
                validAccuracy loaded `shouldBe` validAccuracy tt
                lastModelPath loaded `shouldBe` lastModelPath tt
                tt `shouldBe` loaded
            Nothing -> expectationFailure "Failed to load TrainingTracker"

        

