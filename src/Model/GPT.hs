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


processBatch :: Model -> Tensor -> Tensor
processBatch model input = 
    let
        output = modelForward model input
    in
        output
        


-- computeLoss :: Tensor -> Tensor -> Tensor
-- computeLoss input target = 
--     let
--         loss = F.crossEntropy logits target
--     in
--         loss
  

                      


