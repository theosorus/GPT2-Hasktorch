module Main (main) where

import Lib
import Torch
import NormalLayer
import MLP
import CasualSelfAttention
import Block
import Model

-- nBlock :: Int
-- nBlock = 12

-- nHead :: Int
-- nHead = 12

-- nEmbd :: Int
-- nEmbd = 768

-- blockkSize :: Int
-- blockSize = 1024

-- vocabSize :: Int
-- vocabSize = 30000

-- batchSize :: Int
-- batchSize = 4

main :: IO ()
main = do
    putStrLn "Hello, Haskell!"
    -- configNEmbd configNBlock configVocabSize configNHead configBlockSize 
    model <- modelInit (ModelConfig 256 2 100 2 128)
    print model
    putStrLn "end of main"