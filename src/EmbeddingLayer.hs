{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}




module EmbeddingLayer (
    Config(..)
  , EmbeddingLayer(..)
  , embeddingLayerInit
  , embeddingLayerForward
) where


import Torch.Functional as F
import Torch
import GHC.Generics


data Config = Config
  { 
    configVocabSize :: Int,
    configNEmbd :: Int
  } deriving (Show, Eq)


data EmbeddingLayer = EmbeddingLayer
  { embdDim :: Int
    ,vocabSize :: Int
    ,weight :: Tensor
  } deriving (Generic, Show)


embeddingLayerInit :: Config -> IO EmbeddingLayer
embeddingLayerInit Config{..} = do
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
