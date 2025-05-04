import Test.Hspec
import Block (blockInit, blockForward,Config(..))
import Torch
import CasualSelfAttention (casualSelfAttentionInit, casualSelfAttentionForward, Config(..))
import NormalLayer ( normalLayerInit, normalLayerForward, Config(..))
import MLP (mlpInit, mlpForward, Config(..))
import EmbeddingLayer (embeddingLayerInit, embeddingLayerForward, Config(..))
import Utils (randInt)


--shape output `shouldBe` [batchSize, seqLen, configNEmbd config]


test_block :: Spec
test_block = do
    let batchSize = 1
        seqLen = 10
        embdDim = 64
        nHead = 8
        blockSize = 10
        -- configNEmbd , configNHead  , configBlockSize 
        config = Block.Config embdDim nHead blockSize
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
        config = CasualSelfAttention.Config embdDim nHead blockSize
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
        config = MLP.Config embdDim
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
        config = NormalLayer.Config [embdDim] 1e-5 False
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
        config = EmbeddingLayer.Config vocabSize embdDim
    it "embedding output shape should match [batchSize, seqLen, embdDim]" $ do
        embedding <- embeddingLayerInit config
        input <- randInt [batchSize, seqLen] 0 (vocabSize - 1)
        let output = embeddingLayerForward embedding input
        shape output `shouldBe` [batchSize, seqLen, embdDim]





main :: IO ()

main = hspec $ do
    test_block
    test_casualSelfAttention
    test_mlp
    test_normalLayer
    test_embedding
    
