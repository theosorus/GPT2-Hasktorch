module Train.Training where


import Torch hiding(cos,floor,div)
import Model.GPT
import Data.Dataloader (DataLoader, Batch)
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import Data.LazyDataloader (LazyDataloader, getNextBatch, countBatches,resetDataloader)
import Control.Monad (foldM, when)

import qualified Config as C
import Config (modelDevice)
import Model.Save 

processBatch :: Model -> Batch -> (Tensor,Tensor,Tensor)
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
        acc = computeAccuracy reshapeOutput reshapeY -- (B*T)
    in
        (output,loss,acc)


trainBatch :: (Optimizer opt) => Model -> Batch -> opt -> Double -> IO (Model, Tensor, Tensor)
trainBatch model batch optimizer lr = do
    let (output,loss,acc) = processBatch model batch
    (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
    pure (newModel,loss,output)


processEpochLazy :: (Optimizer opt) => Model -> LazyDataloader -> Maybe LazyDataloader -> opt -> Int -> Int -> Int ->  IO (Model,Maybe LazyDataloader)
processEpochLazy model trainDataloader validDataloader optimizer nbBatch nbEpoch currentEpoch = do
  let
    loop currentModel trainDl validDl currentOptim currentGrad iter = do
      mb <- getNextBatch trainDl
      case mb of
        Nothing -> pure (currentModel,validDl)
        Just (trainBatch, trainDl') -> do
          let (output, loss,acc) = processBatch currentModel trainBatch

          -- accumulate gradients
          newGrads <- if isEmptyGradients currentGrad
                      then pure $ grad' loss $ flattenParameters currentModel
                      else pure $ accumulateGradients currentGrad (grad' loss $ flattenParameters currentModel)

          -- update model parameters if needed
          (finalModel, finalGrad, finalOptim) <-
            if (iter + 1) `mod` C.gradientAccumulationStep == 0 then do
              let lr = getLearningRate (iter + 1 + (nbBatch * currentEpoch-1)) ((nbBatch `div` C.gradientAccumulationStep) * nbEpoch)
              (updatedModel, optState) <- runStep' currentModel currentOptim newGrads (realToFrac lr)
              pure (updatedModel, Gradients [], optState)
            else
              pure (currentModel, newGrads, currentOptim)

          -- evaluate the model and get validation loss if available
          (validLossMaybe, validDlUpdated) <- case validDl of
            Nothing -> pure (Nothing, Nothing)
            Just validDl' -> do
              (validLoss,validAcc, validDl'') <- processTest finalModel validDl'
              pure (Just validLoss, Just validDl'')

          -- print progress with validation loss if available
          when ((iter + 1) `mod` C.printFreq == 0) $
            putStrLn $ 
              "Epoch: " ++ show currentEpoch 
              ++ "/" ++ show nbEpoch
              ++ ", Iteration: " ++
              show (iter + 1) ++ "/"
              ++ show nbBatch
              ++ ", Loss: " ++ show (asValue loss :: Float)
              ++ ", Accuracy: " ++ show (asValue acc :: Float)
              ++ case validLossMaybe of
                   Nothing -> ""
                   Just vLoss -> ", Validation Loss: " ++ show (asValue vLoss :: Float)

          when ((iter + 1) `mod` C.saveFreq == 0) $ do
            let modelPath = getModelPath C.modelName C.modelDir 0 (iter + 1)
            saveModel modelPath finalModel True

          loop finalModel trainDl' validDlUpdated finalOptim finalGrad (iter + 1)

  loop model trainDataloader validDataloader optimizer (Gradients []) 0



processTraining :: (Optimizer opt) => Model -> LazyDataloader -> Maybe LazyDataloader -> opt -> Int -> IO Model
processTraining model trainDataloader initialValidDataloader optimizer nbEpoch = do 
  totalBatches <- countBatches trainDataloader
  putStrLn $ "Total batches: " ++ show totalBatches
  
  (finalModel, _finalValidDlState) <- foldM 
    (\(currentEpochModel, currentEpochValidDl) epoch -> do
      putStrLn $ "Starting epoch " ++ show epoch ++ "/" ++ show nbEpoch
      
      resetTrainDl <- if epoch > 1 
            then resetDataloader trainDataloader
            else pure trainDataloader
        
      (newModelFromEpoch, newValidDlFromEpoch) <- processEpochLazy currentEpochModel resetTrainDl currentEpochValidDl optimizer totalBatches nbEpoch epoch 
      
      pure (newModelFromEpoch, newValidDlFromEpoch) 
    ) 
    (model, initialValidDataloader)
    [1..nbEpoch] 
  putStrLn "Training completed."
  pure finalModel

processTest :: Model -> LazyDataloader -> IO (Tensor,Tensor, LazyDataloader)
processTest model validDl = do
  mb <- getNextBatch validDl
  case mb of
    Nothing -> do 
      newDl <- resetDataloader validDl
      processTest model newDl
    Just (validBatch, validDl') -> do
      let (validOutput, validLoss,validAcc) = processBatch model validBatch
      pure (validLoss,validAcc, validDl')


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
   
  



  