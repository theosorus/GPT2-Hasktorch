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

modelName :: String
modelName = "gpt"

-- Device Configuration
device :: Device
device = Device CPU 0

-- Model Architecture
vocabSize :: Int
vocabSize = 30000

nBlock :: Int
nBlock = 2

nHead :: Int
nHead = 2

nEmbd :: Int
nEmbd = 128

blockSize :: Int
blockSize = 128


-- Training Configuration
batchSize :: Int
batchSize = 4 

epochs :: Int
epochs = 2


saveFreq :: Int
saveFreq = 20

printFreq :: Int
printFreq = 2

-- Optimizer Configuration
betas :: (Float, Float)
betas = (0.9, 0.95)

beta1 :: Float
beta1 = fst betas

beta2 :: Float
beta2 = snd betas

eps :: Double
eps = 1e-8

weightDecay :: Double
weightDecay = 0.1

maxLr :: Double
maxLr = 6e-4

minLr :: Double
minLr = maxLr * 0.1

gradientAccumulationStep :: Int
gradientAccumulationStep = 1

learningRate :: Double
learningRate = 0.001

