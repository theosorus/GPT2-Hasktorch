module ModelTest where

import Test.Hspec
import Model.Block (blockInit, blockForward,BlockConfig(..))
import Torch
import Model.CasualSelfAttention (casualSelfAttentionInit, casualSelfAttentionForward, CasualSelfAttentionConfig(..))
import Model.NormalLayer ( normalLayerInit, normalLayerForward, NormalLayerConfig(..))
import Model.MLP (mlpInit, mlpForward, MLPConfig(..))
import Model.EmbeddingLayer (embeddingLayerInit, embeddingLayerForward, EmbeddingLayerConfig(..))
import Utils (randInt)
import Model.GPT 

import Config (modelDevice)

testBlock :: Spec
testBlock = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = BlockConfig embdDim nHead blockSize
    it "blockForward output shape should match [batchSize, seqLen, embdDim]" $ do
        block <- blockInit config
        rawinput <- randIO' [batchSize, seqLen, embdDim]
        let input = toDevice modelDevice rawinput
        let output = blockForward block input
        shape output `shouldBe` [batchSize, seqLen, embdDim]


testCasualSelfAttention :: Spec
testCasualSelfAttention = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = CasualSelfAttentionConfig embdDim nHead blockSize
    it "casualSelfAttention output shape should match [batchSize, seqLen, embdDim]" $ do
        block <- casualSelfAttentionInit config
        rawinput <- randIO' [batchSize, seqLen, embdDim]
        let input = toDevice modelDevice rawinput
        let output = casualSelfAttentionForward block input
        shape output `shouldBe` [batchSize, seqLen, embdDim]

testMlp :: Spec
testMlp = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = MLPConfig embdDim
    it "mlp output shape should match [batchSize, seqLen, embdDim]" $ do
        mlp <- mlpInit config
        rawinput <- randIO' [batchSize, seqLen, embdDim]
        let input = toDevice modelDevice rawinput
        let output = mlpForward mlp input
        shape output `shouldBe` [batchSize, seqLen, embdDim]


testNormalLayer :: Spec
testNormalLayer = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = NormalLayerConfig [embdDim] 1e-5 False
    it "normalLayer output shape should match [batchSize, seqLen, embdDim]" $ do
        normalLayer <- normalLayerInit config
        rawinput <- randIO' [batchSize, seqLen, embdDim]
        let input = toDevice modelDevice rawinput
        let output = normalLayerForward normalLayer input
        shape output `shouldBe` [batchSize, seqLen, embdDim]


testEmbedding :: Spec
testEmbedding = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        vocabSize = 1000
        -- vocabSize , embdDim
        config = EmbeddingLayerConfig vocabSize embdDim
    it "embedding output shape should match [batchSize, seqLen, embdDim]" $ do
        embedding <- embeddingLayerInit config
        input <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        let output = embeddingLayerForward embedding input
        shape output `shouldBe` [batchSize, seqLen, embdDim]


testModel :: Spec
testModel = do
    let batchSize = 1
        seqLen = 10
        embdDim = 256
        nHead = 8
        blockSize = 128
        vocabSize = 100
        nBlock = 2
        -- configNEmbd configNBlock configVocabSize configNHead configBlockSize 
        config = ModelConfig embdDim nBlock vocabSize nHead blockSize
    it "model output shape should match [batchSize, seqLen, vocabSize]" $ do
        model <- modelInit config
        input <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        let output = modelForward model input
        shape output `shouldBe` [batchSize, seqLen, vocabSize]

testAccuracy :: Spec
testAccuracy = do
    let batchSize = 16
        seqLen = 10
        vocabSize = 100
    it "computeAccuracy output should match [] (scalar)" $ do
        output <- randIO' [batchSize, seqLen, vocabSize] 
        target <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        let acc = computeAccuracy output target
        shape acc `shouldBe` [] -- scalar


testComputeLoss :: Spec
testComputeLoss = do
    let batchSize = 16
        seqLen = 10
        vocabSize = 100
    it "computeLoss output should match [] (scalar)" $ do
        output <- randIO' [batchSize* seqLen , vocabSize] 
        target <- randInt [batchSize * seqLen] 0 (vocabSize - 1)
        let loss = computeCrossEntropyLoss output target
        shape loss `shouldBe` [] -- scalar



