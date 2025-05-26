module Train.Training where

import Torch
import Model.GPT
import Data.Dataloader (DataLoader, Batch)
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import Control.Monad (foldM)
import qualified Config as C
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
        reshapeY = FI.reshape y [-1] -- (B*T)
        loss = computeCrossEntropyLoss reshapeOutput reshapeY -- (B*T)
    in
        (output,loss)


trainBatch :: (Optimizer opt) => Model -> Batch -> opt -> Double -> IO (Model, Tensor, Tensor)
trainBatch model batch optimizer lr = do
    let (output,loss) = processBatch model batch
    (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
    pure (newModel,loss,output)



processEpoch :: (Optimizer opt) => Model -> DataLoader -> opt -> Double  -> IO Model
processEpoch model dataloader optimizer lr = do
    result <- foldM (\(currentModel, currentGrad,currentOptim, iter) batch -> do
        
        -- forward pass
        let (output, loss) = processBatch currentModel batch 

        -- gradient accumulation
        newGrads <- if isEmptyGradients currentGrad then 
            pure $ grad' loss $ flattenParameters currentModel
          else
            pure $ accumulateGradients currentGrad (grad' loss $ flattenParameters currentModel)
        
        -- update model parameters
        (finalModel, finalGrad, finalOptim) <- if (iter + 1) `mod` C.gradientAccumulationStep == 0 then do
          (updatedModel, optState) <- runStep' currentModel optimizer newGrads (realToFrac lr)
          pure (updatedModel, Gradients [],optState)
        else
          pure (currentModel, newGrads, currentOptim)
        
        


        -- Save , print , eval
        if (iter + 1) `mod` C.printFreq == 0 then do
            putStrLn $ show (iter + 1) ++ "/" ++ show (length dataloader)  ++ ", Loss: " ++ show (loss)
        else
            pure ()


        if (iter + 1) `mod` C.saveFreq == 0 then do
          let modelPath = getModelPath C.modelName C.modelDir 0 (iter + 1)
          saveModel modelPath finalModel True
        else 
          pure ()

        pure (finalModel, finalGrad,finalOptim, iter + 1)
      ) (model, Gradients [], optimizer,0) dataloader

    let (finalModel, _, _,_) = result
    pure finalModel

accumulateGradients :: Gradients -> Gradients -> Gradients
accumulateGradients (Gradients currentGradTensor) (Gradients newGradTensor) = Gradients $ zipWith (+) currentGradTensor newGradTensor

isEmptyGradients :: Gradients -> Bool
isEmptyGradients (Gradients ts) = null ts