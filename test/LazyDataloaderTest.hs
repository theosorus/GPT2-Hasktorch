module LazyDataloaderTest where

import Test.Hspec

import Data.LazyDataloader
import Data.File (loadWordsJson)
import Data.Preprocess (wordToIndexFactory)



testCountBatches :: IO ()

testCountBatches = do 
    let testFilePath = "data/test/small_text.txt"
    it "countBatches should return the correct number of batches" $ do
        wordlst <- loadWordsJson vocabTestPath
        let wti = wordToIndexFactory wordlst
            vs  = length wordlst
            bbs = 2048    -- octets lus à chaque appel
            tbs = 512    -- tokens retournés / batch
        count <- countBatches dl
        count `shouldBe` 2 -- Assuming we have enough tokens for 2 batches of size 10





