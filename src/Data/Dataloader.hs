module Data.Dataloader where


import Torch
import Torch.Functional as F
import Torch.Functional.Internal as FI
import GHC.Generics


type DataLoader = [(Tensor, Tensor)]
type Batch = (Tensor, Tensor)


-- input Tensor  :[N]


chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = let (chunk, rest) = splitAt n xs in chunk : chunksOf n rest


-- | Construit un DataLoader à partir d’un tensor 1-D de tokens de longueur N.
createDataLoader
  :: Int     -- ^ batchSize
  -> Int     -- ^ seqLen
  -> Tensor  -- ^ tokens de forme [N]
  -> DataLoader
createDataLoader batchSize seqLen tokens = 
  let 
    -- Get tensor size as Int
    tokenSize = Torch.size 0 tokens 
    
    -- Créer les entrées (x): tranches de longueur seqLen
    createInputs = 
      [ FI.slice tokens 0 i (i + seqLen) 1
      | i <- [0 .. tokenSize - seqLen - 1] ]
    
    -- Créer les cibles (y): tranches de longueur seqLen décalées de 1
    createTargets = 
      [ FI.slice tokens 0 (i + 1) (i + seqLen + 1) 1
      | i <- [0 .. tokenSize - seqLen - 1] ]
    
    -- Combiner les entrées et les cibles en un dataset
    dataset = zip createInputs createTargets
    
    -- Découper en batchs de taille batchSize
    batches = filter (\chunk -> length chunk == batchSize) (chunksOf batchSize dataset)
    
    -- Traiter chaque batch pour créer des tensors empilés
    processBatch batch =
      let inputs = map fst batch
          targets = map snd batch
          inputBatch = Torch.stack (Dim 0) inputs    -- [B, T]
          targetBatch = Torch.stack (Dim 0) targets  -- [B, T]
      in (inputBatch, targetBatch)
    
  in map processBatch batches