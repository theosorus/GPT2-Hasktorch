{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Data.LazyDataloader where

import Torch (size)
import System.IO (Handle, openFile, IOMode(..), hClose, hSetBinaryMode)
import qualified Data.ByteString.Lazy as B
import qualified Torch.Functional.Internal as FI
import GHC.Generics (Generic)
import Torch (Tensor, asTensor)

import Data.Preprocess (preprocess)  

    

data LazyDataloader = LazyDataloader
    { wordToIndex    :: B.ByteString -> Int
    , vocabSize      :: Int
    , currentIndex   :: Int
    , tokenBlockSize :: Int
    , byteBlockSize  :: Int
    , handle         :: Handle
    , tokenBuffer    :: [Int]
    } deriving (Generic)

countBatches :: LazyDataloader -> IO Int
countBatches dl = go 0 dl
  where
    go :: Int -> LazyDataloader -> IO Int
    go cnt d = do
      mb <- getNextBlock d
      case mb of
        Just (_, d') -> go (cnt + 1) d'
        Nothing      -> return cnt


createBatch :: Tensor -> -> (Tensor,Tensor)



getNextBatch :: LazyDataloader -> IO (Maybe ((Tensor,Tensor), LazyDataloader))
getNextBatch dl = do
  mb <- getNextBlock dl
  case mb of
    Just (batch, dl') -> return (Just (createBatch batch, dl'))
    Nothing           -> return Nothing

initLazyDataloader ::
  FilePath ->
  Int ->    -- byteBlockSize
  Int ->    -- tokenBlockSize
  (B.ByteString -> Int) ->
  Int ->
  IO LazyDataloader
initLazyDataloader path bbs tbs wti vs = do
  h <- openFile path ReadMode
  hSetBinaryMode h True
  return LazyDataloader
    { wordToIndex    = wti
    , vocabSize      = vs
    , currentIndex   = 0
    , byteBlockSize  = bbs
    , tokenBlockSize = tbs
    , handle         = h
    , tokenBuffer    = []
    }

fillBuffer :: LazyDataloader -> IO LazyDataloader
fillBuffer dl@LazyDataloader{..}
  | length tokenBuffer >= tokenBlockSize = return dl
  | otherwise = do
      chunk <- B.hGet handle byteBlockSize
      if B.null chunk
        then return dl
        else do
          let newTokens = map wordToIndex (preprocess chunk)
              dl'       = dl { tokenBuffer = tokenBuffer ++ newTokens }
          fillBuffer dl'

getNextBlock :: LazyDataloader -> IO (Maybe (Tensor, LazyDataloader))
getNextBlock dl = do
  dlFilled@LazyDataloader{..} <- fillBuffer dl
  if length tokenBuffer < tokenBlockSize
    then hClose handle >> return Nothing
    else do
      let (blockTokens, rest) = splitAt tokenBlockSize tokenBuffer
          t    = asTensor blockTokens :: Tensor
          dl'  = dlFilled
            { tokenBuffer  = rest
            , currentIndex = currentIndex + tokenBlockSize
            }
      return (Just (t, dl'))




