module LazyDataloaderTest where

import Test.Hspec
import Torch

import Data.LazyDataloader
import Data.File (loadWordsJson)
import Data.Preprocess (wordToIndexFactory)



testCountBatches :: Spec
testCountBatches = do 
    let testFilePath = "data/tests/small_text.txt"
        vocabTestPath = "data/tests/vocab_test.json"
    it "countBatches lazy dataloader should return the correct number of batches" $ do
        wordlst <- loadWordsJson vocabTestPath
        let wti = wordToIndexFactory wordlst
            vs  = length wordlst
            bbs = 2048    
            tbs = 512    
        dl <- initDataloader testFilePath bbs tbs wti vs
        count <- countBatches dl
        count `shouldBe` 1 


testSizeBatches :: Spec
testSizeBatches = do 
    let testFilePath = "data/tests/small_text.txt"
        vocabTestPath = "data/tests/vocab_test.json"
    it "All batches in lazy dataloader should have the good size" $ do
        wordlst <- loadWordsJson vocabTestPath
        let wti = wordToIndexFactory wordlst
            vs  = length wordlst
            bbs = 2048    
            tbs = 512    
        dl <- initDataloader testFilePath bbs tbs wti vs

        let loopTest d = do
              mb <- getNextBlock d
              case mb of
                Just (batch, d') -> do
                  (size 0 batch) `shouldBe` tbs
                  loopTest d'
                Nothing          -> return ()
        loopTest dl





