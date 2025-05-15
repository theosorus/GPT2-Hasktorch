{-# LANGUAGE OverloadedStrings #-}

module Data.File where

import qualified Data.ByteString.Lazy as B
import Data.Aeson
import qualified Data.Map as Map
import Data.Maybe (fromMaybe)
import System.IO
import Data.List.Split (splitOn)

type CharMap = Map.Map String Int


loadJSON :: FilePath -> IO (Maybe CharMap)
loadJSON filePath = do
  content <- B.readFile filePath
  return $ decode content

-- merge.txt
-- a l
-- a r
-- Ġt o
-- Ġ m
-- Ġo f
-- Ġ in
readFileToPairs :: FilePath -> String -> IO [(String, String)]
readFileToPairs filePath delimiter = do
    content <- readFile filePath
    let lines' = lines content         -- Diviser en lignes
        pairs = map (toPair delimiter) lines'   -- Convertir chaque ligne en tuple
    return pairs
  where
    -- Fonction pour convertir une ligne en tuple (s1, s2)
    toPair :: String -> String -> (String, String)
    toPair delim line = 
        case splitOn delim line of
            [first, second] -> (first, second)  -- Cas normal: deux parties
