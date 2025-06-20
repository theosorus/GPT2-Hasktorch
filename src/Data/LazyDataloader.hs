{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Data.LazyDataloader where

import Torch (size,reshape)
import System.IO (Handle, openFile, IOMode(..), hClose, hSetBinaryMode)
import qualified Data.ByteString.Lazy as B
import qualified Torch.Functional.Internal as FI
import qualified Torch.Functional as F
import GHC.Generics (Generic)
import Torch (Tensor, asTensor)

import Data.Preprocess (preprocess)
import Tokenizer.Tokenizer (tokenMain)
import Config (vocabPath, mergesPath)


type Batch = (Tensor, Tensor)

data LazyDataloader = LazyDataloader
    { filePath       :: FilePath
    , vocabSize      :: Int
    , currentIndex   :: Int
    , tokenBlockSize :: Int
    , batchSize      :: Int
    , seqLen         :: Int
    , byteBlockSize  :: Int
    , handle         :: Handle
    , tokenBuffer    :: [Int]
    , totalBatches   :: Int      
    , currentBatch   :: Int      
    } deriving (Generic)

countBatches :: LazyDataloader -> IO Int
countBatches dl = do
  dl' <- internalInitLazyDataloader
           (filePath dl)
           (byteBlockSize dl)
           (batchSize dl)
           (seqLen dl)
           (vocabSize dl)
  let
    go :: Int -> LazyDataloader -> IO Int
    go cnt d = do
      mb <- getNextBlock d
      case mb of
        Just (_, d') -> go (cnt + 1) d'
        Nothing      -> return cnt
  go 0 dl'


createBatch :: Tensor -> Int -> Int -> Batch
createBatch t batchSize seqLen =
  let
    t' = reshape [batchSize, seqLen + 1] t --  [B, T+1]
    x  = FI.narrow_copy t' 1 0 seqLen -- [B, T]
    y  = FI.narrow_copy t' 1 1 seqLen -- [B, T]
  in
    (x, y)

getNextBatch :: LazyDataloader -> IO (Maybe (Batch, LazyDataloader))
getNextBatch dl = do
  mb <- getNextBlock dl
  case mb of
    Just (batch, dl') ->
      let bs = batchSize dl'
          sl = seqLen    dl'
          dl'' = dl'
            { currentBatch = currentBatch dl' + 1 }
      in return (Just (createBatch batch bs sl, dl''))
    Nothing ->
      return Nothing



initLazyDataloader ::
  FilePath ->
  Int ->    -- byteBlockSize
  Int ->    -- batchSize
  Int ->    -- seqLen
  Int ->
  IO LazyDataloader
initLazyDataloader path bbs bs sq vs = do
  dl0 <- internalInitLazyDataloader path bbs bs sq vs
  --n <- countBatches dl0
  let n = 100000
  return $ dl0 { totalBatches = n }



internalInitLazyDataloader ::
  FilePath ->
  Int ->    -- byteBlockSize
  Int ->    -- batchSize
  Int ->    -- seqLen
  Int ->
  IO LazyDataloader
internalInitLazyDataloader path bbs bs sq vs = do
  h <- openFile path ReadMode
  hSetBinaryMode h True

  let dl0 = LazyDataloader
        { filePath       = path
        , vocabSize      = vs
        , currentIndex   = 0
        , byteBlockSize  = bbs
        , batchSize      = bs
        , seqLen         = sq
        , tokenBlockSize = (bs * sq) + bs
        , handle         = h
        , tokenBuffer    = []
        , totalBatches   = 0
        , currentBatch   = 0
        }
  return dl0 





resetDataloader :: LazyDataloader -> IO LazyDataloader
resetDataloader dl@LazyDataloader{..} = do
    hClose handle
    h <- openFile filePath ReadMode
    hSetBinaryMode h True
    return dl
      { handle       = h
      , currentIndex = 0
      , tokenBuffer  = []
      , currentBatch = 0  
      }


fillBuffer :: LazyDataloader -> IO LazyDataloader
fillBuffer dl@LazyDataloader{..}
  | length tokenBuffer >= tokenBlockSize = return dl
  | otherwise = do
      chunk <- B.hGet handle byteBlockSize
      if B.null chunk
        then return dl
        else do
          tokens <- tokenMain (B.toStrict chunk) vocabPath mergesPath
          let safeTokens   = map (\i -> if i < 0 || i >= vocabSize then 0 else i) tokens
              dl'          = dl { tokenBuffer = tokenBuffer ++ safeTokens }
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




