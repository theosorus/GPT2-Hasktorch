{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module NormalLayer (
    Config(..)
  , NormalLayer(..)
  , normalLayerInit
  , normalLayerForward
) where


import GHC.Generics
import Torch
import Torch.Functional.Internal as FI

data Config = Config
    { normalized_shape_config :: [Int]
    , eps_config :: Double
    , cudnn_enable_config :: Bool
    } deriving (Show, Eq)

data NormalLayer = NormalLayer
  { normalized_shape :: [Int]
  , weight :: Tensor
  , bias :: Tensor
  , eps :: Double
  , cudnn_enable :: Bool
  } deriving (Generic, Show)


normalLayerInit :: Config -> IO NormalLayer
normalLayerInit Config{..} = do
  

  weight <- randIO' normalized_shape_config
  bias <- randIO' normalized_shape_config
     
  return NormalLayer
    { 
        normalized_shape = normalized_shape_config
    , weight = weight
    , bias = bias
    , eps = eps_config
    , cudnn_enable = cudnn_enable_config
    }

normalLayerForward
  :: NormalLayer
  -> Tensor
  ->  Tensor
normalLayerForward NormalLayer{..} input = FI.layer_norm input normalized_shape weight bias eps cudnn_enable