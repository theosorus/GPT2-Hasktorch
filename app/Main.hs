module Main (main) where

import Torch

import Model.GPT
import Utils (randInt)
import Data.Dataloader 
import qualified Config as C
import Train.Training


main :: IO ()
main = do
    putStrLn "Hello, Haskell!"




