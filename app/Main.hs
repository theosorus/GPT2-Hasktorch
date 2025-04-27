module Main (main) where

import Lib
import Torch

main :: IO ()
main = do
    putStrLn "Hello, Haskell!"
    x  <- randIO' [10,10]
    putStrLn $ "Tensor: " ++ show x
