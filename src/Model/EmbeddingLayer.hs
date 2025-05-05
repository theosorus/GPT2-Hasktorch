{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


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
    ,weight :: Tensor
  } deriving (Generic, Show)


embeddingLayerInit :: EmbeddingLayerConfig -> IO EmbeddingLayer
embeddingLayerInit EmbeddingLayerConfig{..} = do
  weight <- randIO' [configVocabSize, configNEmbd]
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
embeddingLayerForward EmbeddingLayer{..} input = F.embedding' weight input