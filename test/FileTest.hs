module FileTest where

import Test.Hspec
import Data.File 
import qualified Data.Map as Map


testLoadVocab :: Spec
testLoadVocab = do
    let dataPath = "data/tokenizer/vocab.json"

        
    -- exected size 50 000
    it "test the size of the vocab" $ do
        result <- loadJSON dataPath
        case result of
            Just charMap -> Map.size charMap `shouldBe` 50257
            Nothing -> expectationFailure "Failed to load JSON file"


testLoadMergeTxt :: Spec
testLoadMergeTxt = do
    let mergePath = "data/tokenizer/merges.txt"
        delimiter = " "
    it "test the size of the merge file" $ do
        result <- readFileToPairs mergePath delimiter
        length result `shouldBe` 50000

testOneItemMerge :: Spec
testOneItemMerge = do
    let mergePath = "data/tokenizer/merges.txt"
        delimiter = " "
    it "test the size of the merge file" $ do
        result <- readFileToPairs mergePath delimiter
        let (s1,s2) = head result        
        s1 `shouldBe` "Ġ"
        s2 `shouldBe` "t"  

        
        

        
