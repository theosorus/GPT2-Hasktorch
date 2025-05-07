{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Model.NormalLayer (
    NormalLayerConfig(..)
  , NormalLayer(..)
  , normalLayerInit
  , normalLayerForward
) where


import GHC.Generics
import Torch
import Torch.Functional.Internal as FI

data NormalLayerConfig = NormalLayerConfig
    { normalized_shape_config :: [Int]
    , eps_config :: Double
    , cudnn_enable_config :: Bool
    } deriving (Show, Eq)

data NormalLayer = NormalLayer
  { normalized_shape :: [Int]
  , weight :: Parameter
  , bias :: Parameter
  , eps :: Double
  , cudnn_enable :: Bool
  } deriving (Generic, Show,Parameterized)


normalLayerInit :: NormalLayerConfig -> IO NormalLayer
normalLayerInit NormalLayerConfig{..} = do
  wTensor <- randIO' normalized_shape_config
  bTensor <- randIO' normalized_shape_config
  weight <- makeIndependent wTensor
  bias <- makeIndependent bTensor
  return NormalLayer
    { normalized_shape = normalized_shape_config
    , weight = weight
    , bias = bias
    , eps = eps_config
    , cudnn_enable = cudnn_enable_config
    }

normalLayerForward
  :: NormalLayer
  -> Tensor
  -> Tensor
normalLayerForward NormalLayer{..} input =
  FI.layer_norm input normalized_shape (toDependent weight) (toDependent bias) eps cudnn_enable