module Config where 

import Torch

-- Paths
dataDir :: String
dataDir = "data"

vocabPath :: String
vocabPath = ""

encoding :: String
encoding = "utf-8"

modelDir :: String
modelDir = "models"

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

-- Unified Configuration
data Config = Config
  { configVocabSize :: Int
  , configNBlock :: Int
  , configNHead :: Int
  , configNEmbd :: Int
  , configBatchSize :: Int
  , configEpochs :: Int
  , configEvalFreq :: Int
  , configSaveFreq :: Int
  , configBetas :: (Double, Double)
  , configEps :: Double
  , configWeightDecay :: Double
  , configMaxLr :: Double
  , configMinLr :: Double
  } deriving (Show, Eq)

-- Default configuration
defaultConfig :: Config
defaultConfig = Config
  { configVocabSize = vocabSize
  , configNBlock = nBlock
  , configNHead = nHead
  , configNEmbd = nEmbd
  , configBatchSize = batchSize
  , configEpochs = epochs
  , configEvalFreq = evalFreq
  , configSaveFreq = saveFreq
  , configBetas = betas
  , configEps = eps
  , configWeightDecay = weightDecay
  , configMaxLr = maxLr
  , configMinLr = minLr
  }