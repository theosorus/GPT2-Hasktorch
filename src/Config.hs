module Config where 

import Torch

-- Device Configuration
device :: Device
device = Device CPU 0

-- Model Architecture
vocabSize :: Int
vocabSize = 30000

nBlock :: Int
nBlock = 12

nHead :: Int
nHead = 12

nEmbd :: Int
nEmbd = 768

-- Training Configuration
batchSize :: Int
batchSize = 4 

epochs :: Int
epochs = 2

evalFreq :: Int
evalFreq = 10

saveFreq :: Int
saveFreq = 50

-- Optimizer Configuration
betas :: (Double, Double)
betas = (0.9, 0.95)

eps :: Double
eps = 1e-8

weightDecay :: Double
weightDecay = 0.1

maxLr :: Double
maxLr = 6e-4

minLr :: Double
minLr = maxLr * 0.1



data Config = Config
  { vocabSize :: Int
  , nBlock :: Int
  , nHead :: Int
  , nEmbd :: Int
  , batchSize :: Int
  , epochs :: Int
  , evalFreq :: Int
  , saveFreq :: Int
  , betas :: (Double, Double)
  , eps :: Double
  , weightDecay :: Double
  , maxLr :: Double
  , minLr :: Double
  } deriving (Show, Eq)


defaultConfig :: Config
defaultConfig = Config
  { vocabSize = vocabSize
  , nBlock = nBlock
  , nHead = nHead
  , nEmbd = nEmbd
  , batchSize = batchSize
  , epochs = epochs
  , evalFreq = evalFreq
  , saveFreq = saveFreq
  , betas = betas
  , eps = eps
  , weightDecay = weightDecay
  , maxLr = maxLr
  , minLr = minLr
  }






