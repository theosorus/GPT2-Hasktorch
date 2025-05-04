{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module MLP (
    Config(..)
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




data Config = Config
  { configNEmbd :: Int
  } deriving (Show, Eq)

data MLP = MLP
  { fcLayer :: Linear
  , projLayer :: Linear
  , nEmbd :: Int
  } deriving (Generic, Show)


mlpInit :: Config -> IO MLP
mlpInit Config{..} = do
  
  fcLayer <- sample (LinearSpec configNEmbd (4 * configNEmbd))
  projLayer <- sample (LinearSpec (4 * configNEmbd) configNEmbd)
     
  return MLP
    { fcLayer = fcLayer
    , projLayer = projLayer
    , nEmbd = configNEmbd
    }


mlpForward :: MLP -> Tensor -> Tensor
mlpForward MLP{..} x = let
 fc = FI.gelu $ NN.linear fcLayer x
 proj = NN.linear projLayer fc
  in 
   proj