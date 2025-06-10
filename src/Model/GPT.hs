{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}



module Model.GPT where 



import GHC.Generics
import Torch
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import Data.Aeson          (ToJSON, FromJSON)
import Torch.NN as NN

import Model.EmbeddingLayer 
import Model.Block
import Model.NormalLayer
import Config (modelDevice)

import Data.Dataloader (DataLoader,Batch)


data ModelConfig = ModelConfig
  { configNEmbd :: Int
  , configNBlock :: Int
  , configVocabSize :: Int
  , configNHead :: Int
  , configBlockSize :: Int
  } deriving (Show, Eq,Generic)

instance ToJSON ModelConfig
instance FromJSON ModelConfig

data Model = Model
  { wte :: EmbeddingLayer,
    wpe :: EmbeddingLayer,
    h :: [Block],
    ln_f :: NormalLayer,
    lm_head :: Linear,
    nEmbd :: Int,
    nHead :: Int,
    nBlock :: Int,
    blockSize :: Int,
    vocabSize :: Int
  } deriving (Generic,Parameterized)


instance Show Model where
  show Model{..} =
    "Model { nEmbd = "   ++ show nEmbd   ++
    ", nHead = "   ++ show nHead   ++
    ", nBlock = "  ++ show nBlock  ++
    ", vocabSize = " ++ show vocabSize ++
    ", blockSize = "  ++ show blockSize ++ " }"

modelInit :: ModelConfig -> IO Model
modelInit c@ModelConfig{..} = do
    wte <- embeddingLayerInit (EmbeddingLayerConfig configVocabSize configNEmbd)
    wpe <- embeddingLayerInit (EmbeddingLayerConfig configBlockSize configNEmbd)

    ln_f <- normalLayerInit (NormalLayerConfig [configNEmbd] 1e-5 False)
    lm_head <- sample (LinearSpec configNEmbd configVocabSize)
    h <- mapM (const $ blockInit (BlockConfig configNEmbd configNHead configBlockSize)) [1..configNBlock]
    
    let model = Model
          { wte = wte
          , wpe = wpe
          , h = h
          , ln_f = ln_f
          , lm_head = lm_head
          , nEmbd = configNEmbd
          , nHead = configNHead
          , blockSize = configBlockSize
          , vocabSize = configVocabSize
          , nBlock = configNBlock
          }
  
    return (toDevice modelDevice model)



modelForward :: Model -> Tensor -> Tensor
modelForward Model{..} input = 
    let
        inputDevice = toDevice modelDevice input -- (B, T)
        pos = toDevice modelDevice $ toDType Int64 (arange' 0 ((shape inputDevice) !! 1) 1) -- (T)
        pos_emnb = embeddingLayerForward wpe pos --  (T, n_embd)
        input_emnb = embeddingLayerForward wte inputDevice -- (B, T, n_embd)
        embd = input_emnb + (FI.unsqueeze pos_emnb 0) -- (B, T, n_embd)

        outputBlock = foldl (\acc block -> blockForward block acc) embd h -- (B, T, n_embd)

        outputBlockNormalized = normalLayerForward ln_f outputBlock -- (B, T, n_embd)
        logits = NN.linear lm_head outputBlockNormalized -- (B, T, vocab_size)
    in 
        logits


computeAccuracy :: Tensor -> Tensor -> Tensor
computeAccuracy predictions targets = 
  let correctPredictions = eq (FI.argmax predictions (-1) False) targets
      numCorrect = F.sumAll correctPredictions
      total = asTensor $ numel targets
  in numCorrect / total


        

-- cross_entropy_loss
-- :: Tensor	self
-- -> Tensor	target
-- -> Tensor	weight
-- -> Int reduction
-- -> Int ignore_index
-- -> Double	label_smoothing
-- -> Tensor	 

computeCrossEntropyLoss :: Tensor -> Tensor -> Tensor
computeCrossEntropyLoss output target =
  let
    outputD = toDevice modelDevice output
    targetD = toDevice modelDevice target
    weight  = toDevice modelDevice $ ones' [last (shape outputD)]
    loss    = FI.cross_entropy_loss outputD targetD weight 1 (-100) 0.0
  in
    loss



                      


