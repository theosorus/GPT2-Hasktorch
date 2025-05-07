{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}



module Model.GPT where 



import GHC.Generics
import Torch
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import Torch.NN as NN

import Model.EmbeddingLayer 
import Model.Block
import Model.NormalLayer

import Data.Dataloader (DataLoader,Batch)


data ModelConfig = ModelConfig
  { configNEmbd :: Int
    , configNBlock :: Int
  , configVocabSize :: Int
  , configNHead :: Int
  , configBlockSize :: Int
  } deriving (Show, Eq)

data Model = Model
  { wte :: EmbeddingLayer,
  wpe :: EmbeddingLayer,
  h :: [Block],
  ln_f :: NormalLayer,
  lm_head :: Linear,
    nEmbd :: Int,
    nHead :: Int,
    nLayer :: Int,
    nBlock :: Int,
    blockSize :: Int,
    vocabSize :: Int

  } deriving (Generic)


instance Show Model where
  show Model{..} =
    "Model { nEmbd = "   ++ show nEmbd   ++
    ", nHead = "   ++ show nHead   ++
    ", nBlock = "  ++ show nBlock  ++
    ", vocabSize = " ++ show vocabSize ++
    ", blockSize = "  ++ show blockSize ++ " }"

modelInit :: ModelConfig -> IO Model
modelInit ModelConfig{..} = do


    wte <- embeddingLayerInit (EmbeddingLayerConfig configVocabSize configNEmbd)
    wpe <- embeddingLayerInit (EmbeddingLayerConfig configBlockSize configNEmbd)

    ln_f <- normalLayerInit (NormalLayerConfig [configNEmbd] 1e-5 False)
    lm_head <- sample (LinearSpec configNEmbd configVocabSize)
    h <- mapM (const $ blockInit (BlockConfig configNEmbd configNHead configBlockSize)) [1..configNBlock]
    return Model
      { wte = wte
      , wpe = wpe
      , h = h
      , ln_f = ln_f
      , lm_head = lm_head
      , nEmbd = configNEmbd
      , nHead = configNHead
      , blockSize = configBlockSize
      , vocabSize = configVocabSize
      ,nBlock = configNBlock
      }


modelForward :: Model -> Tensor -> Tensor
modelForward Model{..} input = 
    let
        pos = toDType Int64 (arange' 0 ((shape input) !! 1) 1) -- (T)
        pos_emnb = embeddingLayerForward wpe pos --  (T, n_embd)
        input_emnb = embeddingLayerForward wte input -- (B, T, n_embd)
        embd = input_emnb + (FI.unsqueeze pos_emnb 0) -- (B, T, n_embd)

        outputBlock = foldl (\acc block -> blockForward block acc) embd h -- (B, T, n_embd)

        outputBlockNormalized = normalLayerForward ln_f outputBlock -- (B, T, n_embd)
        logits = NN.linear lm_head outputBlockNormalized -- (B, T, vocab_size)
    in 
        logits


processBatch :: Model -> Batch -> (Tensor,Tensor)
processBatch model (x,y) = 
  -- x :  (B, T) 
  -- y :  (B, T)
  -- output : (B, T, vocab_size)
    let
        output = modelForward model x
        nbLogits = last $ shape output 
        reshapeOutput = FI.reshape output [-1,nbLogits] -- (B*T, vocab_size)
        reshapeY = FI.reshape y [-1] -- (B*T)
        loss = computeCrossEntropyLoss reshapeOutput reshapeY -- (B*T)
    in
        (output,loss)



        

-- cross_entropy_lossSource
-- :: Tensor	self
-- -> Tensor	target
-- -> Tensor	weight
-- -> Int reduction
-- -> Int ignore_index
-- -> Double	label_smoothing
-- -> Tensor	 

computeCrossEntropyLoss :: Tensor -> Tensor -> Tensor
computeCrossEntropyLoss output target = 
  -- output : (B*T, vocab_size)
  -- target : (B*T)
    let
      weight = ones' [last (shape output)]
      loss = FI.cross_entropy_loss output target weight 1 (-100) 0.0

    in
      loss
  



                      


