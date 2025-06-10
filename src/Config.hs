module Config where 

import Torch


-- Paths
dataDir :: String
dataDir = "data"

vocabPath :: String
vocabPath = "data/tokenizer/vocab.json"

mergesPath :: String
mergesPath = "data/tokenizer/merges.txt"

byteBlockSize :: Int
byteBlockSize = 1024 * 1024 * 10 * 10   -- 10 MB

encoding :: String
encoding = "utf-8"

modelDir :: String
modelDir = "models"

modelName :: String
modelName = "gpt"

trainingTrackerPath :: String
trainingTrackerPath = "training_tracker.json"

-- Device Configuration
modelDevice :: Device
modelDevice = Device CUDA 0

-- Model Architecture
vocabSize :: Int
vocabSize =  1  

nBlock :: Int
nBlock = 1 -- 12

nHead :: Int
nHead = 1 -- 12

nEmbd :: Int
nEmbd = 1

blockSize :: Int
blockSize = 10 -- 1024


-- Training Configuration
batchSize :: Int
batchSize =1

epochs :: Int
epochs = 2


saveFreq :: Int
saveFreq = 500

printFreq :: Int
printFreq = 1

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
gradientAccumulationStep = 5

learningRate :: Double
learningRate = 0.001

