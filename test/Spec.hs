import Test.Hspec



import DataTest
import ModelTest
import TrainingTest
import FileTest


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
    

    -- DATA
    testDataloaderLength
    testDataloaderSizeFirstItem
    testDataloaderSizeLastItem

    -- TRAINING
    testProcessBatch
    testTrainBatch
    testProcessEpoch
    

    -- FILE
    testLoadJson
    
