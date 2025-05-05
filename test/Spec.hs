import Test.Hspec



import DataTest
import ModelTest


--shape output `shouldBe` [batchSize, seqLen, configNEmbd config]







main :: IO ()

main = hspec $ do
    test_block
    test_casualSelfAttention
    test_mlp
    test_normalLayer
    test_embedding
    test_model
    test_dataloader_length
    test_dataloader_size_first_item
    test_dataloader_size_last_item
    
