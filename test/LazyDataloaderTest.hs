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
        dl <- initLazyDataloader testFilePath bbs tbs wti vs
        count <- countBatches dl
        count `shouldBe` 1 


testSizeBlock :: Spec
testSizeBlock= do 
    let testFilePath = "data/tests/small_text.txt"
        vocabTestPath = "data/tests/vocab_test.json"
    it "All block in lazy dataloader should have the good size" $ do
        wordlst <- loadWordsJson vocabTestPath
        let wti = wordToIndexFactory wordlst
            vs  = length wordlst
            bbs = 2048    
            tbs = 512    
        dl <- initLazyDataloader testFilePath bbs tbs wti vs

        let loopTest d = do
              mb <- getNextBlock d
              case mb of
                Just (batch, d') -> do
                  (size 0 batch) `shouldBe` tbs
                  loopTest d'
                Nothing          -> return ()
        loopTest dl


testSizeBatch :: Spec
testSizeBatch= do 
    let testFilePath = "data/tests/small_text.txt"
        vocabTestPath = "data/tests/vocab_test.json"
    it "All batches in lazy dataloader should have the good size" $ do
        wordlst <- loadWordsJson vocabTestPath
        let wti = wordToIndexFactory wordlst
            vs  = length wordlst
            bbs = 2048    
            tbs = 512    
        dl <- initLazyDataloader testFilePath bbs tbs wti vs
        
        batch <- getNextBatch dl
        case batch of
          Just ((x, y), dl') -> do
            putStrLn $ "x shape: " ++ show (shape x)
            putStrLn $ "y shape: " ++ show (shape y)
          Nothing -> fail "Expected to get at least one batch"




-- testBatch :: Spec
-- testBatch= do 
--     let testFilePath = "data/tests/small_text.txt"
--         vocabTestPath = "data/tests/vocab_test.json"
--     it "tes deep batch" $ do
--         wordlst <- loadWordsJson vocabTestPath
--         let wti = wordToIndexFactory wordlst
--             vs  = length wordlst
--             bbs = 1024
--             tbs = 64   
--         dl1 <- initLazyDataloader testFilePath bbs tbs wti vs
        
--         Just (x1, dl2) <- getNextBlock dl1
--         putStrLn $ " \n x1" ++ show x1 ++ "\n"
--         putStrLn $ "buffer " ++ show (tokenBuffer dl2) ++ "\n"
--         putStrLn $ "buffer size " ++ show (length (tokenBuffer dl2)) ++ "\n"

--         Just (x2, dl3) <- getNextBlock dl2
--         putStrLn $ "x2" ++ show x2 ++ "\n"
--         putStrLn $ "buffer " ++ show (tokenBuffer dl3) ++ "\n"
--         putStrLn $ "buffer size " ++ show (length (tokenBuffer dl3)) ++ "\n"

--         Just (x3, dl4) <- getNextBlock dl3
--         putStrLn $ "x3" ++ show x3 ++ "\n"
--         putStrLn $ "buffer " ++ show (tokenBuffer dl4) ++ "\n"
--         putStrLn $ "buffer size " ++ show (length (tokenBuffer dl4)) ++ "\n"

--         Just (x4, dl5) <- getNextBlock dl4
--         putStrLn $ "x4" ++ show x4 ++ "\n"
--         putStrLn $ "buffer " ++ show (tokenBuffer dl5) ++ "\n"
--         putStrLn $ "buffer size " ++ show (length (tokenBuffer dl5)) ++ "\n"
    








