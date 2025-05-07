{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}


module Model.EmbeddingLayer (
    EmbeddingLayerConfig(..)
  , EmbeddingLayer(..)
  , embeddingLayerInit
  , embeddingLayerForward
) where


import Torch.Functional as F
import Torch
import GHC.Generics


data EmbeddingLayerConfig = EmbeddingLayerConfig
  { 
    configVocabSize :: Int,
    configNEmbd :: Int
  } deriving (Show, Eq)


data EmbeddingLayer = EmbeddingLayer
  { embdDim :: Int
    ,vocabSize :: Int
    ,weight :: Parameter
  } deriving (Generic, Show, Parameterized)


embeddingLayerInit :: EmbeddingLayerConfig -> IO EmbeddingLayer
embeddingLayerInit EmbeddingLayerConfig{..} = do
  wTensor <- randIO' [configVocabSize, configNEmbd]
  weight <- makeIndependent wTensor
  return EmbeddingLayer
    { 
        embdDim = configNEmbd
      , vocabSize = configVocabSize
      , weight = weight
    }

embeddingLayerForward
  :: EmbeddingLayer
  -> Tensor
  -> Tensor
embeddingLayerForward EmbeddingLayer{..} input = F.embedding' (toDependent weight) input