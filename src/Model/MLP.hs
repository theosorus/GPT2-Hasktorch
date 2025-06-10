{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Model.MLP (
    MLPConfig(..)
  , MLP(..)
  , mlpInit
  , mlpForward
) where

import GHC.Generics
import Torch
import Torch.NN as NN
import Torch.Functional as F
import Torch.Functional.Internal as FI
import Torch.TensorFactories
import Torch.TensorOptions
import Control.Monad (when)
import Config (modelDevice)

data MLPConfig = MLPConfig
  { configNEmbd :: Int
  } deriving (Show, Eq)

data MLP = MLP
  { fcLayer   :: Linear
  , projLayer :: Linear
  , nEmbd     :: Int
  } deriving (Generic, Show, Parameterized)

mlpInit :: MLPConfig -> IO MLP
mlpInit MLPConfig{..} = do
  fcLayer   <- sample (LinearSpec configNEmbd (4 * configNEmbd))
  projLayer <- sample (LinearSpec (4 * configNEmbd) configNEmbd)
  let mlp = MLP
        { fcLayer   = fcLayer
        , projLayer = projLayer
        , nEmbd     = configNEmbd
        }
  return (toDevice modelDevice mlp)

mlpForward :: MLP -> Tensor -> Tensor
mlpForward MLP{..} x =
  let
    fcOut   = FI.gelu (NN.linear fcLayer x) "none"
    projOut = NN.linear projLayer fcOut
  in
    projOut