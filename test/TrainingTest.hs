module TrainingTest where

import Training
import Torch

import Test.Hspec
import Utils (randInt)
import Model.GPT


testProcessBatch :: Spec
testProcessBatch = do
    let batchSize = 16
        seqLen = 10
        vocabSize = 100
        embdDim = 256
        nHead = 8
        blockSize = 128
        nBlock = 2
        -- configNEmbd configNBlock configVocabSize configNHead configBlockSize 
        config = ModelConfig embdDim nBlock vocabSize nHead blockSize
    it "processBatch output shape should match [batchSize, seqLen, vocabSize] and loss should match []" $ do
        model <- modelInit config
        x <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        y <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        let input = (x, y)
        let (output,loss) = processBatch model input
        shape output `shouldBe` [batchSize, seqLen, vocabSize]
        shape loss `shouldBe` [] -- scalar



testTrainBatch :: Spec
testTrainBatch = do
    let batchSize = 16
        seqLen = 10
        vocabSize = 100
        embdDim = 256
        nHead = 8
        blockSize = 128
        nBlock = 2
        lr = 0.001
        optimizer = GD
        -- configNEmbd configNBlock configVocabSize configNHead configBlockSize 
        config = ModelConfig embdDim nBlock vocabSize nHead blockSize
    it "trainBatch must return newModel output : [batchSize, seqLen, vocabSize] and loss : []" $ do
        
        -- Initialize the model
        model <- modelInit config

        -- create data
        x <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        y <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        let input = (x, y)

        -- Train the model on the batch
        (newModel,loss,output) <- trainBatch model input optimizer lr

        -- check the new model
        newModel `shouldSatisfy` (const True :: Model -> Bool)

        -- Check the output shapes
        shape output `shouldBe` [batchSize, seqLen, vocabSize]
        shape loss `shouldBe` [] -- scalar
        
        