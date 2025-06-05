module Train.Training where


import Torch hiding(cos,floor,div)
import Model.GPT
import Data.Dataloader (DataLoader, Batch)
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import Data.LazyDataloader (LazyDataloader, getNextBatch, resetDataloader, totalBatches)
import Control.Monad (foldM, when)
import Data.Maybe (fromMaybe)

import qualified Config as C
import Config (modelDevice)
import Model.Save 
import Train.TrainingTracker



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




processEpochLazy 
  :: (Optimizer opt)
  => Model
  -> LazyDataloader
  -> Maybe LazyDataloader
  -> opt
  -> Int        -- ^ nbEpoch
  -> Int        -- ^ currentEpoch
  -> TrainingTracker
  -> IO (Model, Maybe LazyDataloader, TrainingTracker)
processEpochLazy model trainDataloader validDataloader optimizer nbEpoch epochNum initTracker = do
  let
    loop currentModel trainDl validDl currentOptim currentGrad iter curTracker = do
      mb <- getNextBatch trainDl
      case mb of
        Nothing -> pure (currentModel, validDl, curTracker)
        Just (trainBatch, trainDl') -> do


          -- on run le forward
          let (output, loss, acc) = processBatch currentModel trainBatch


          -- on met à jour le tracker pour le train
          let curTracker1 = curTracker
                { currentEpoch     = epochNum  -- Use the parameter instead of the field name
                , currentBatch     = iter + 1
                , trainLoss        = trainLoss curTracker ++ [asValue loss :: Float]
                , trainAccuracy    = trainAccuracy curTracker ++ [asValue acc :: Float]
                }


          -- accumulation des gradients
          newGrads <- if isEmptyGradients currentGrad
                      then pure $ grad' loss $ flattenParameters currentModel
                      else pure $ accumulateGradients currentGrad (grad' loss $ flattenParameters currentModel)


          -- apply/update step si nécessaire
          (finalModel, finalGrad, finalOptim) <- 
            if (iter + 1) `mod` C.gradientAccumulationStep == 0 then do
              let lr = getLearningRate (iter + 1 + ((totalBatches trainDl) * epochNum - 1))
                                       ((totalBatches trainDl `div` C.gradientAccumulationStep) * nbEpoch)
              (updM, optS) <- runStep' currentModel currentOptim newGrads (realToFrac lr)
              pure (updM, Gradients [], optS)
            else
              pure (currentModel, newGrads, currentOptim)


          -- évaluation validation si existante
          (validLossMaybe, validDlUpdated, curTracker2) <- case validDl of
            Nothing -> pure (Nothing, Nothing, curTracker1)
            Just vdl -> do
              (vLoss, vAcc, vdl'') <- processTest finalModel vdl
              let t' = curTracker1
                    { validLoss     = validLoss t' ++ [asValue vLoss :: Float]
                    , validAccuracy = validAccuracy t' ++ [asValue vAcc :: Float]
                    }
              pure (Just vLoss, Just vdl'', t')


          -- affichage & save inchangés...
          when ((iter + 1) `mod` C.printFreq == 0) $
            putStrLn $ 
              "Epoch: " ++ show epochNum
              ++ "/" ++ show nbEpoch
              ++ ", Iteration: " ++
              show (iter + 1) ++ "/"
              ++ show (totalBatches trainDl')
              ++ ", Loss: " ++ show (asValue loss :: Float)
              ++ ", Accuracy: " ++ show (asValue acc :: Float)
              ++ case validLossMaybe of
                   Nothing -> ""
                   Just vLoss -> ", Validation Loss: " ++ show (asValue vLoss :: Float)



          when ((iter + 1) `mod` C.saveFreq == 0) $ do
            let modelPath = getModelPath C.modelName C.modelDir 0 (iter + 1)
            saveModel modelPath finalModel True
          

          
          loop finalModel trainDl' validDlUpdated finalOptim finalGrad (iter + 1) curTracker2

  loop model trainDataloader validDataloader optimizer (Gradients []) 0 initTracker

processTraining 
  :: (Optimizer opt)
  => Model
  -> LazyDataloader
  -> Maybe LazyDataloader
  -> opt
  -> Int
  -> Maybe TrainingTracker
  -> IO (Model, TrainingTracker)
processTraining model trainDataloader initialValidDataloader optimizer nbEpoch mTracker = do

  let tracker0 = fromMaybe initialTrainingTracker mTracker

  (finalModel, _validDl, finalTracker) <- foldM
    (\(curModel, curValidDl, curTracker) epoch -> do
       putStrLn $ "Starting epoch " ++ show epoch ++ "/" ++ show nbEpoch
       resetTrainDl <- if epoch > 1 then resetDataloader trainDataloader else pure trainDataloader
       processEpochLazy curModel resetTrainDl curValidDl optimizer nbEpoch epoch curTracker
    )
    (model, initialValidDataloader, tracker0)
    [1..nbEpoch]

  putStrLn "Training completed."
  pure (finalModel, finalTracker)



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





