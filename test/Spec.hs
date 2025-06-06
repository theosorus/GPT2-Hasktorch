import Test.Hspec



import DataTest
import ModelTest
import TrainingTest
import FileTest
import SaveTest
import LazyDataloaderTest
import TrainingTrackerTest

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
    testAccuracy
    

    -- DATA
    testDataloaderLength
    testDataloaderSizeFirstItem
    testDataloaderSizeLastItem

    -- TRAINING
    testProcessBatch
    testTrainBatch
    testProcessEpochLazy
    testProcessTraining
    testProcessTrainingWithValid
    

    -- FILE
    testLoadVocab
    testLoadMergeTxt
    testOneItemMerge

    -- SAVE
    testSaveLoadModel

    -- LAZY DATALOADER
    testCountBatches
    testSizeBlock
    testSizeBatch
    testResetLazyDataloader

    -- TRAINING TRACKER
    trainingTrackerSaveAndLoad
    
    
