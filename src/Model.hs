{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}



module Model where 



import GHC.Generics
import Torch
import qualified Torch.Functional as F

import EmbeddingLayer 
import Block
import NormalLayer


data ModelConfig = ModelConfig
  { configNEmbd :: Int
    , configNBlock :: Int
  , configVocabSize :: Int
  , configNHead :: Int
  , configBlockSize :: Int
  } deriving (Show, Eq)

data Model = Model
  { wte :: EmbeddingLayer,
  wpe :: EmbeddingLayer,
  h :: [Block],
  ln_f :: NormalLayer,
  lm_head :: Linear,
    nEmbd :: Int,
    nHead :: Int,
    nLayer :: Int,
    nBlock :: Int,
    blockSize :: Int,
    vocabSize :: Int

  } deriving (Generic)


instance Show Model where
  show Model{..} =
    "Model { nEmbd = "   ++ show nEmbd   ++
    ", nHead = "   ++ show nHead   ++
    ", nBlock = "  ++ show nBlock  ++
    ", vocabSize = " ++ show vocabSize ++
    ", blockSize = "  ++ show blockSize ++ " }"

modelInit :: ModelConfig -> IO Model
modelInit ModelConfig{..} = do


    wte <- embeddingLayerInit (EmbeddingLayerConfig configVocabSize configNEmbd)
    wpe <- embeddingLayerInit (EmbeddingLayerConfig configBlockSize configNEmbd)

    ln_f <- normalLayerInit (NormalLayerConfig [configNEmbd] 1e-5 False)
    lm_head <- sample (LinearSpec configNEmbd configVocabSize)
    h <- mapM (const $ blockInit (BlockConfig configNEmbd configNHead configBlockSize)) [1..configNBlock]
    return Model
      { wte = wte
      , wpe = wpe
      , h = h
      , ln_f = ln_f
      , lm_head = lm_head
      , nEmbd = configNEmbd
      , nHead = configNHead
      , blockSize = configBlockSize
      , vocabSize = configVocabSize
      ,nBlock = configNBlock
      }

  

                      
