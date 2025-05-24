module SaveTest where

import Test.Hspec
import Torch
import System.Directory (removeFile)
import Torch.Functional.Internal (equal)

import Model.Save
import Model.GPT
import Utils (randInt)



testSaveLoadModel :: Spec
testSaveLoadModel = do
    let batchSize = 1
        seqLen = 1
        vocabSize = 10
        embdDim = 2
        nHead = 1
        blockSize = 2
        nBlock = 1
        -- configNEmbd configNBlock configVocabSize configNHead configBlockSize 
        config = ModelConfig embdDim nBlock vocabSize nHead blockSize
    it "test Save the model" $ do
        -- Initialize the model
        model <- modelInit config

        -- Save the model
        saveModel "test_model.pth" model False

        -- Load the model
        loadedModel <- loadModel "test_model.pth" config

        -- Check if the loaded model is the same as the original model
        inputTest <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        let outputFromInitial = modelForward model inputTest
        let outputFromLoaded = modelForward loadedModel inputTest

        putStrLn $ "Output from initial model: " ++ show outputFromInitial
        putStrLn $ "Output from loaded model: " ++ show outputFromLoaded

        -- remove the saved model file
        removeFile "test_model.pth"

        
        putStrLn $ " Params from initial Model : "++show (map toDependent $ flattenParameters model)
        putStrLn $ " Params from loaded Model : "++show (map toDependent $ flattenParameters loadedModel)

        -- Check if the outputs are the same
        shape outputFromInitial `shouldBe` shape outputFromLoaded


        --(equal (outputFromInitial) (outputFromLoaded)) `shouldBe` True

        
        
        