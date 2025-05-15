{-# LANGUAGE OverloadedStrings #-}

module Data.File
  ( loadJSON
  ) where

import qualified Data.ByteString.Lazy as B
import Data.Aeson
import qualified Data.Map as Map
import Data.Maybe (fromMaybe)

type CharMap = Map.Map String Int


loadJSON :: FilePath -> IO (Maybe CharMap)
loadJSON filePath = do
  content <- B.readFile filePath
  return $ decode content