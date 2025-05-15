module FileTest where

import Test.Hspec
import Data.File 
import qualified Data.Map as Map


testLoadJson :: Spec
testLoadJson = do
    let dataPath = "data/tokenizer/vocab.json"

        
    -- exected size 50 000
    it "test the size of the json extracted" $ do
        result <- loadJSON dataPath
        case result of
            Just charMap -> Map.size charMap `shouldBe` 50257
            Nothing -> expectationFailure "Failed to load JSON file"
        
