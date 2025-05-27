{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Data.LazyDataloader where

import System.IO (Handle, openFile, IOMode(..), hClose, hSetBinaryMode)
import qualified Data.ByteString.Lazy as B
import GHC.Generics (Generic)
import Torch (Tensor, asTensor)

import Data.Preprocess (preprocess)  

    

data Dataloader = Dataloader
    { wordToIndex    :: B.ByteString -> Int
    , vocabSize      :: Int
    , currentIndex   :: Int
    , tokenBlockSize :: Int            
    , byteBlockSize  :: Int            
    , handle         :: Handle
    , tokenBuffer    :: [B.ByteString] 
    } deriving (Generic)

countBatches :: Dataloader -> IO Int
countBatches dl = go 0 dl
  where
    go :: Int -> Dataloader -> IO Int
    go cnt d = do
      mb <- getNextBlock d
      case mb of
        Just (_, d') -> go (cnt + 1) d'
        Nothing      -> return cnt


getNextBlock :: Dataloader -> IO (Maybe (Tensor, Dataloader))
getNextBlock dl = do
  dlFilled@Dataloader{..} <- fillBuffer dl
  if length tokenBuffer < tokenBlockSize
    then hClose handle >> return Nothing
    else do
      let (blockTokens, rest) = splitAt tokenBlockSize tokenBuffer
          idxs = map wordToIndex blockTokens
          t    = asTensor idxs :: Tensor
          dl'  = dlFilled
            { tokenBuffer  = rest
            , currentIndex = currentIndex + tokenBlockSize
            }
      return (Just (t, dl'))

initDataloader ::
  FilePath ->
  Int ->    -- byteBlockSize
  Int ->    -- tokenBlockSize
  (B.ByteString -> Int) ->
  Int ->
  IO Dataloader
initDataloader path bbs tbs wti vs = do
  h <- openFile path ReadMode
  hSetBinaryMode h True
  return Dataloader
    { wordToIndex    = wti
    , vocabSize      = vs
    , currentIndex   = 0
    , byteBlockSize  = bbs
    , tokenBlockSize = tbs
    , handle         = h
    , tokenBuffer    = []
    }

fillBuffer :: Dataloader -> IO Dataloader
fillBuffer dl@Dataloader{..}
  | length tokenBuffer >= tokenBlockSize = return dl
  | otherwise = do
      chunk <- B.hGet handle byteBlockSize
      if B.null chunk
        then return dl
        else do
          let ws  = preprocess chunk
              dl' = dl { tokenBuffer = tokenBuffer ++ ws }
          fillBuffer dl'


    

