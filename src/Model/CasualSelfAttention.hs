{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


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
  } deriving (Generic, Show)



createCausalMask :: Int -> Tensor
createCausalMask seqLen =
  FI.reshape triangle [1, 1, seqLen, seqLen]
  where
    triangle = FI.tril onesMatrix 0
    onesMatrix = ones' [seqLen, seqLen]



casualSelfAttentionInit :: CasualSelfAttentionConfig -> IO CasualSelfAttention
casualSelfAttentionInit CasualSelfAttentionConfig{..} = do
  
  -- Vérification que nEmbd est divisible par nHead
  when (configNEmbd `mod` configNHead /= 0) $
    error "configNEmbd doit être divisible par configNHead"
  
  -- Initialisation des couches linéaires
  cAttn <- sample (LinearSpec configNEmbd (3 * configNEmbd))
  cProj <- sample (LinearSpec configNEmbd configNEmbd)

  let attentionBias = createCausalMask configBlockSize
                      
  return CasualSelfAttention
    { cAttn = cAttn
    , cProj = cProj
    , nHead = configNHead
    , nEmbd = configNEmbd
    , attentionBias = attentionBias
    }


scaledDotProductAttention :: Tensor -> Tensor -> Tensor -> Maybe Tensor -> Maybe Float -> Tensor
scaledDotProductAttention q k v mask dropout = 
  let
    -- Calcul des scores d'attention et mise à l'échelle
    scaleFactor = FI.sqrt (fromIntegral $ last $ shape k)
    scores = FI.div (FI.matmul q (FI.transpose k (-2) (-1))) scaleFactor
  
    
    -- Application du masque causal si fourni
    maskedScores = case mask of
      Just m -> scores * m + (1.0 - m) * (-1e10)
      Nothing -> scores
    
    -- Softmax sur les scores
    tyype = dtype q
    weights = FI.softmax maskedScores (-1) tyype
    
    -- Application du dropout si spécifié
    droppedWeights = case dropout of
       --Just rate -> dropout2d weights rate True
       Just rate -> weights
       Nothing -> weights
    
    -- Valeurs pondérées
    output = FI.matmul weights v
  in
    output


casualSelfAttentionForward :: CasualSelfAttention -> Tensor -> Tensor
casualSelfAttentionForward CasualSelfAttention{..} x = let
  -- Obtenir les dimensions du batch, sequence length et embedding size
  shapes = shape x
  batchSize = head shapes 
  seqLen = shapes !! 1
  embedSize = shapes !! 2
  headSize = Prelude.div embedSize nHead  -- Division d'entiers, pas de tenseurs
  
  -- Projeter l'entrée en query, key, value
  -- Supposons que c'est une opération linéaire:
  projected = NN.linear cAttn x
  
  -- Diviser en 3 parties selon la dimension 2
  qkvList = FI.chunk projected 3 2 
  q = head qkvList 
  k = qkvList !! 1
  v = qkvList !! 2
  
  -- -- Reshape pour multi-têtes d'attention
  q' = FI.transpose (F.view [batchSize, seqLen, nHead, headSize] q ) 1 2
  k' = FI.transpose (F.view [batchSize, seqLen, nHead, headSize] k) 1 2
  v' = FI.transpose (F.view [batchSize, seqLen, nHead, headSize] v ) 1 2

  
  currentMask = createCausalMask seqLen
  -- -- Attention avec masque causal
  att = scaledDotProductAttention q' k' v' (Just currentMask) Nothing
  
  -- -- Reshape pour la sortie
  y1 = FI.transpose att 1 2
  y2 = contiguous y1
  y = F.view [batchSize, seqLen, embedSize] y2 
  
  -- -- Projection finale
  output = NN.linear cProj y
  in 
   output