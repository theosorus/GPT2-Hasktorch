import Test.Hspec



import DataTest
import ModelTest


--shape output `shouldBe` [batchSize, seqLen, configNEmbd config]







main :: IO ()

main = hspec $ do

    -- MODEL
    testBlock
    testCasualSelfAttention
    testMlp
    testNormalLayer
    testEmbedding
    testModel
    testComputeLoss
    testProcessBatch

    -- DATA
    testDataloaderLength
    testDataloaderSizeFirstItem
    testDataloaderSizeLastItem
    
    
