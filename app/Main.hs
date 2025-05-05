module Main (main) where

import Model.GPT



main :: IO ()
main = do
    putStrLn "Hello, Haskell!"
    -- configNEmbd configNBlock configVocabSize configNHead configBlockSize 
    model <- modelInit (ModelConfig 256 2 100 2 128)
    print model
    putStrLn "end of main"