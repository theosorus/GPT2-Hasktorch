{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module Block (
    Block.Config(..)
  , Block(..)
  , blockInit
    , blockForward
) where

import GHC.Generics
import Torch
import Torch.NN as NN
import Torch.Functional as F
import Torch.Functional.Internal as FI

import MLP
import CasualSelfAttention
import NormalLayer




data Config = Config
    { embdDim_config :: Int,

    nHead_config :: Int,
    blockSize_config :: Int
    } deriving (Show, Eq)

data Block = Block
  { embdDim :: Int
    ,ln1 :: NormalLayer
    , attn :: CasualSelfAttention
    , ln2 :: NormalLayer
    , mlp :: MLP
  } deriving (Generic, Show)

blockInit :: Block.Config -> IO Block
blockInit Block.Config{..} = do
  ln1 <- normalLayerInit (NormalLayer.Config [embdDim_config] 1e-5  False )
  ln2 <- normalLayerInit (NormalLayer.Config [embdDim_config] 1e-5  False )

  attn <- casualSelfAttentionInit (CasualSelfAttention.Config embdDim_config nHead_config blockSize_config)
  mlp <- mlpInit (MLP.Config embdDim_config)


     
  return Block
    { embdDim = embdDim_config
    , ln1 = ln1
    , attn = attn
    , ln2 = ln2
    , mlp = mlp

    }

blockForward
  :: Block
  -> Tensor
  -> Tensor
blockForward Block{..} x = let
    x1 = normalLayerForward ln1 x
    x2 = casualSelfAttentionForward attn x1
    x3 = normalLayerForward ln2 x2
    x4 = mlpForward mlp x3
    in 
     x4