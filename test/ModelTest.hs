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

test_block :: Spec
test_block = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = BlockConfig embdDim nHead blockSize
    it "blockForward output shape should match [batchSize, seqLen, embdDim]" $ do
        block <- blockInit config
        input <- randIO' [batchSize, seqLen, embdDim]
        let output = blockForward block input
        shape output `shouldBe` [batchSize, seqLen, embdDim]


test_casualSelfAttention :: Spec
test_casualSelfAttention = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = CasualSelfAttentionConfig embdDim nHead blockSize
    it "casualSelfAttention output shape should match [batchSize, seqLen, embdDim]" $ do
        block <- casualSelfAttentionInit config
        input <- randIO' [batchSize, seqLen, embdDim]
        let output = casualSelfAttentionForward block input
        shape output `shouldBe` [batchSize, seqLen, embdDim]

test_mlp :: Spec
test_mlp = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = MLPConfig embdDim
    it "mlp output shape should match [batchSize, seqLen, embdDim]" $ do
        mlp <- mlpInit config
        input <- randIO' [batchSize, seqLen, embdDim]
        let output = mlpForward mlp input
        shape output `shouldBe` [batchSize, seqLen, embdDim]


test_normalLayer :: Spec
test_normalLayer = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = NormalLayerConfig [embdDim] 1e-5 False
    it "normalLayer output shape should match [batchSize, seqLen, embdDim]" $ do
        normalLayer <- normalLayerInit config
        input <- randIO' [batchSize, seqLen, embdDim]
        let output = normalLayerForward normalLayer input
        shape output `shouldBe` [batchSize, seqLen, embdDim]


test_embedding :: Spec
test_embedding = do
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


test_model :: Spec
test_model = do
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


test_computeLoss :: Spec
test_computeLoss = do
    let batchSize = 16
        seqLen = 10
        vocabSize = 100
    it "computeLoss output should match [] (scalar)" $ do
        output <- randIO' [batchSize* seqLen , vocabSize] 
        target <- randInt [batchSize * seqLen] 0 (vocabSize - 1)
        let loss = computeCrossEntropyLoss output target
        shape loss `shouldBe` [] -- scalar


test_processBatch :: Spec
test_processBatch = do
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
