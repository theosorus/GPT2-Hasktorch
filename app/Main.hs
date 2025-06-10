module Main (main) where

import Torch

import Model.GPT
import Utils (randInt)
import Data.Dataloader 
import qualified Config as C
import Train.Training

import qualified Config as C
import Model.GPT (ModelConfig(..), modelInit)
import Train.Training (processTraining)
import Data.LazyDataloader (initLazyDataloader)


main :: IO ()
main = do
    putStrLn "Hello, Haskell!"
    trainDl <- initLazyDataloader "data/valid.txt" C.byteBlockSize C.batchSize C.blockSize C.vocabSize
    putStrLn "Training dataloader initialized."
    validDl <- initLazyDataloader "data/valid.txt" C.byteBlockSize C.batchSize C.blockSize C.vocabSize
    putStrLn "Validation dataloader initialized."


    let config = ModelConfig C.nEmbd C.nBlock C.vocabSize C.nHead C.blockSize
    model <- modelInit config

    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters model)


    (finalModel,finalTracker) <- processTraining model trainDl (Just validDl) optimizer C.epochs Nothing

    putStrLn "Training completed!"




