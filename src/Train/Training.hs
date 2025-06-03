module Train.Training where

import Torch hiding( cos,floor,div)
import Model.GPT
import Data.Dataloader (DataLoader, Batch)
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import Data.LazyDataloader (LazyDataloader, getNextBatch, countBatches,resetDataloader)
import Control.Monad (foldM, when)

import qualified Config as C
import Config (modelDevice)
import Model.Save 

processBatch :: Model -> Batch -> (Tensor,Tensor)
processBatch model (x,y) = 
  -- x :  (B, T) 
  -- y :  (B, T)
  -- output : (B, T, vocab_size)
    let
        output = modelForward model x
        nbLogits = last $ shape output 
        reshapeOutput = FI.reshape output [-1,nbLogits] -- (B*T, vocab_size)

        yDevice = toDevice modelDevice y

        reshapeY = FI.reshape yDevice [-1] -- (B*T)
        loss = computeCrossEntropyLoss reshapeOutput reshapeY -- (B*T)
    in
        (output,loss)


trainBatch :: (Optimizer opt) => Model -> Batch -> opt -> Double -> IO (Model, Tensor, Tensor)
trainBatch model batch optimizer lr = do
    let (output,loss) = processBatch model batch
    (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
    pure (newModel,loss,output)


processEpochLazy :: (Optimizer opt) => Model -> LazyDataloader -> opt -> Int -> Int -> Int -> IO Model
processEpochLazy model dataloader optimizer nbBatch nbEpoch currentEpoch = do
  let
    loop currentModel dl currentOptim currentGrad iter = do
      mb <- getNextBatch dl
      case mb of
        Nothing -> pure currentModel
        Just (batch, dl') -> do
          let (output, loss) = processBatch currentModel batch

          -- accumulate gradients
          newGrads <- if isEmptyGradients currentGrad
                      then pure $ grad' loss $ flattenParameters currentModel
                      else pure $ accumulateGradients currentGrad (grad' loss $ flattenParameters currentModel)

          (finalModel, finalGrad, finalOptim) <-
            if (iter + 1) `mod` C.gradientAccumulationStep == 0 then do
              let lr = getLearningRate (iter + 1 + (nbBatch * currentEpoch-1)) ((nbBatch `div` C.gradientAccumulationStep) * nbEpoch)
              (updatedModel, optState) <- runStep' currentModel currentOptim newGrads (realToFrac lr)
              pure (updatedModel, Gradients [], optState)
            else
              pure (currentModel, newGrads, currentOptim)

          when ((iter + 1) `mod` C.printFreq == 0) $
            putStrLn $
              show (iter + 1) ++ "/"
              ++ show nbBatch
              ++ ", Loss: " ++ show loss

          when ((iter + 1) `mod` C.saveFreq == 0) $ do
            let modelPath = getModelPath C.modelName C.modelDir 0 (iter + 1)
            saveModel modelPath finalModel True

          loop finalModel dl' finalOptim finalGrad (iter + 1)

  loop model dataloader optimizer (Gradients []) 0



processTraining :: (Optimizer opt) => Model -> LazyDataloader -> opt -> Int ->  IO Model
processTraining model dataloader optimizer nbEpoch = do 
  totalBatches <- countBatches dataloader
  putStrLn $ "Total batches: " ++ show totalBatches
  
  finalModel <- foldM 
    (\currentModel epoch -> do
      putStrLn $ "Starting epoch " ++ show epoch ++ "/" ++ show nbEpoch
      
      dl <- if epoch > 1 
            then resetDataloader dataloader
            else pure dataloader
        
      newModel <- processEpochLazy currentModel dl optimizer totalBatches nbEpoch epoch 
      
      pure newModel
    ) 
    model 
    [1..nbEpoch]
  putStrLn "Training completed."
  pure finalModel
  
accumulateGradients :: Gradients -> Gradients -> Gradients
accumulateGradients (Gradients currentGradTensor) (Gradients newGradTensor) = Gradients $ zipWith (+) currentGradTensor newGradTensor

isEmptyGradients :: Gradients -> Bool
isEmptyGradients (Gradients ts) = null ts


getLearningRate :: Int -> Int -> Float  
getLearningRate iter totalIter = 
  let
    warmUpStep = floor (0.1 * fromIntegral totalIter)
    cosine_decay = 0.5 * (1 + cos (pi * fromIntegral (iter - warmUpStep) / fromIntegral (totalIter - warmUpStep)))
  in
    case () of  
      _ | iter < warmUpStep -> realToFrac C.maxLr * (fromIntegral (iter + 1) / fromIntegral warmUpStep)
        | iter > totalIter -> realToFrac C.minLr
        | otherwise -> realToFrac C.minLr + cosine_decay * (realToFrac $ C.maxLr - C.minLr)
   
  



  