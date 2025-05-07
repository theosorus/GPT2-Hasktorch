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


-- | Builds a DataLoader from a 1-D tensor of tokens of length N.
createDataLoader
  :: Int     -- ^ batchSize
  -> Int     -- ^ seqLen
  -> Tensor  -- ^ tokens of shape [N]
  -> DataLoader
createDataLoader batchSize seqLen tokens = 
  let 
    -- Get tensor size as Int
    tokenSize = Torch.size 0 tokens 
    
    -- Create inputs (x): slices of length seqLen
    createInputs = 
      [ FI.slice tokens 0 i (i + seqLen) 1
      | i <- [0 .. tokenSize - seqLen - 1] ]
    
    -- Create targets (y): slices of length seqLen offset by 1
    createTargets = 
      [ FI.slice tokens 0 (i + 1) (i + seqLen + 1) 1
      | i <- [0 .. tokenSize - seqLen - 1] ]
    
    -- Combine inputs and targets into a dataset
    dataset = zip createInputs createTargets
    
    -- Split into batches of size batchSize
    batches = filter (\chunk -> length chunk == batchSize) (chunksOf batchSize dataset)
    
    -- Process each batch to create stacked tensors
    processBatch batch =
      let inputs = map fst batch
          targets = map snd batch
          inputBatch = Torch.stack (Dim 0) inputs    -- [B, T]
          targetBatch = Torch.stack (Dim 0) targets  -- [B, T]
      in (inputBatch, targetBatch)
    
  in map processBatch batches