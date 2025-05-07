{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}


module Model.CasualSelfAttention (
  CasualSelfAttentionConfig(..)
  , CasualSelfAttention(..)
  , casualSelfAttentionInit
  , casualSelfAttentionForward
) where



import GHC.Generics
import Torch
import Torch.NN as NN
import Torch.Functional as F
import Torch.Functional.Internal as FI

import Torch.TensorFactories
import Torch.TensorOptions

import Control.Monad (when)

data CasualSelfAttentionConfig = CasualSelfAttentionConfig
  { configNEmbd :: Int
  , configNHead :: Int
  , configBlockSize :: Int
  } deriving (Show, Eq)

data CasualSelfAttention = CasualSelfAttention
  { cAttn :: Linear
  , cProj :: Linear
  , nHead :: Int
  , nEmbd :: Int
  , attentionBias :: Tensor
  } deriving (Generic, Show,Parameterized)



createCausalMask :: Int -> Tensor
createCausalMask seqLen =
  FI.reshape triangle [1, 1, seqLen, seqLen]
  where
  triangle = FI.tril onesMatrix 0
  onesMatrix = ones' [seqLen, seqLen]



casualSelfAttentionInit :: CasualSelfAttentionConfig -> IO CasualSelfAttention
casualSelfAttentionInit CasualSelfAttentionConfig{..} = do
  
  -- Check that nEmbd is divisible by nHead
  when (configNEmbd `mod` configNHead /= 0) $
    error "configNEmbd must be divisible by configNHead"
  
  -- Initialize linear layers
  cAttn <- sample (LinearSpec configNEmbd (3 * configNEmbd))
  cProj <- sample (LinearSpec configNEmbd configNEmbd)

  let attentionBias = createCausalMask configBlockSize
  return $ CasualSelfAttention
    { cAttn = cAttn
    , cProj = cProj
    , nHead = configNHead
    , nEmbd = configNEmbd
    , attentionBias = attentionBias
    }


scaledDotProductAttention :: Tensor -> Tensor -> Tensor -> Maybe Tensor -> Maybe Float -> Tensor
scaledDotProductAttention q k v mask dropout = 
  let
  -- Calculate attention scores and scaling
  scaleFactor = FI.sqrt (fromIntegral $ last $ shape k)
  scores = FI.div (FI.matmul q (FI.transpose k (-2) (-1))) scaleFactor
  
  
  -- Apply causal mask if provided
  maskedScores = case mask of
    Just m -> scores * m + (1.0 - m) * (-1e10)
    Nothing -> scores
  
  -- Softmax on scores
  tyype = dtype q
  weights = FI.softmax maskedScores (-1) tyype
  
  -- Apply dropout if specified
  droppedWeights = case dropout of
     --Just rate -> dropout2d weights rate True
     Just rate -> weights
     Nothing -> weights
  
  -- Weighted values
  output = FI.matmul weights v
  in
  output


casualSelfAttentionForward :: CasualSelfAttention -> Tensor -> Tensor
casualSelfAttentionForward CasualSelfAttention{..} x = let
  -- Get batch dimensions, sequence length and embedding size
  shapes = shape x
  batchSize = head shapes 
  seqLen = shapes !! 1
  embedSize = shapes !! 2
  headSize = Prelude.div embedSize nHead  -- Integer division, not tensor division
  
  -- Project input to query, key, value
  -- Assuming this is a linear operation:
  projected = NN.linear cAttn x
  
  -- Split into 3 parts along dimension 2
  qkvList = FI.chunk projected 3 2 
  q = head qkvList 
  k = qkvList !! 1
  v = qkvList !! 2
  
  -- Reshape for multi-head attention
  q' = FI.transpose (F.view [batchSize, seqLen, nHead, headSize] q ) 1 2
  k' = FI.transpose (F.view [batchSize, seqLen, nHead, headSize] k) 1 2
  v' = FI.transpose (F.view [batchSize, seqLen, nHead, headSize] v ) 1 2

  
  currentMask = createCausalMask seqLen
  -- Attention with causal mask
  att = scaledDotProductAttention q' k' v' (Just currentMask) Nothing
  
  -- Reshape for output
  y1 = FI.transpose att 1 2
  y2 = contiguous y1
  y = F.view [batchSize, seqLen, embedSize] y2 
  
  -- Final projection
  output = NN.linear cProj y
  in 
   output