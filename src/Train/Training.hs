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
import ML.Exp.Chart   (drawLearningCurve)



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




processEpochLazy model trainDataloader validDataloader optimizer nbEpoch epochNum initTracker = do
  let
    loop currentModel trainDl validDl currentOptim currentGrad curTracker = do
      mb <- getNextBatch trainDl
      case mb of
        Nothing -> pure (currentModel, validDl, curTracker)
        Just (trainBatch, trainDl') -> do

          let currentIteration = currentBatch curTracker + 1
          let (_output, loss, acc) = processBatch currentModel trainBatch

          -- accumulation des gradients & update step…
          newGrads <- if isEmptyGradients currentGrad
                      then pure $ grad' loss $ flattenParameters currentModel
                      else pure $ accumulateGradients currentGrad (grad' loss $ flattenParameters currentModel)

          -- update model if enough gradients accumulated
          (finalModel, finalGrad, finalOptim) <- 
            if (currentIteration) `mod` C.gradientAccumulationStep == 0 then do
              let lr = getLearningRate (currentIteration + ((totalBatches trainDl) * epochNum - 1))
                                       ((totalBatches trainDl `div` C.gradientAccumulationStep) * nbEpoch)
              (updM, optS) <- runStep' currentModel currentOptim newGrads (realToFrac lr)
              pure (updM, Gradients [], optS)
            else
              pure (currentModel, newGrads, currentOptim)

          -- validation
          (validLossMaybe, validAccMaybe, validDlUpdated) <- case validDl of
            Nothing  -> pure (Nothing, Nothing, Nothing)
            Just vdl -> do
              (vLoss, vAcc, vdl'') <- processTest finalModel vdl
              pure (Just vLoss, Just vAcc, Just vdl'')

          

          let
            condSave  = currentIteration `mod` C.saveFreq == 0
            modelPath = if condSave
                        then Just (getModelPath C.modelName C.modelDir epochNum currentIteration)
                        else Nothing
            updatedTracker = updateTracker curTracker epochNum currentIteration loss acc validLossMaybe validAccMaybe modelPath

          -- affichage
          when (currentIteration `mod` C.printFreq == 0) $ do
            printTrainingInfo
              epochNum
              nbEpoch
              currentIteration
              (totalBatches trainDl')
              loss
              acc
              validLossMaybe
              validAccMaybe
            drawCurvesFromTracker updatedTracker

          -- sauvegarde
          when condSave $ do
            let Just path = modelPath
            saveModel path finalModel True
            saveTrainingTracker C.trainingTrackerPath updatedTracker

          loop finalModel trainDl' validDlUpdated finalOptim finalGrad updatedTracker
  loop model trainDataloader validDataloader optimizer (Gradients []) initTracker



printTrainingInfo :: Int -> Int -> Int -> Int        
                  -> Tensor     
                  -> Tensor     
                  -> Maybe Tensor  
                  -> Maybe Tensor  
                  -> IO ()
printTrainingInfo epochNum nbEpoch currentIteration totalIter trainLoss trainAcc validLossMaybe validAccMaybe = 
  putStrLn $ 
    "Epoch: " ++ show epochNum
    ++ "/" ++ show nbEpoch
    ++ ", Iter : " ++
    show currentIteration ++ "/"
    ++ show totalIter
    ++ ", train_loss: " ++ show (asValue trainLoss :: Float)
    ++ ", train_acc: " ++ show (asValue trainAcc :: Float)
    ++ case validLossMaybe of
         Nothing -> ""
         Just vLoss -> ", valid_loss: " ++ show (asValue vLoss :: Float)
    ++ case validAccMaybe of
         Nothing -> ""
         Just vAcc -> ", valid_acc: " ++ show (asValue vAcc :: Float)



drawCurvesFromTracker :: TrainingTracker -> IO ()
drawCurvesFromTracker tracker = do
  let tLoss = trainLoss tracker
      vLoss = validLoss tracker
      tAcc  = trainAccuracy tracker
      vAcc  = validAccuracy tracker
  drawLearningCurve "output/loss.png" "Training Losses"
    [("Train Loss", tLoss), ("Valid Loss", vLoss)]
  drawLearningCurve "output/acc.png" "Training Accuracy"
    [("Train Acc",  tAcc), ("Valid Acc",  vAcc)]



updateTracker
  :: TrainingTracker
  -> Int              -- ^ epochNum
  -> Int              -- ^ currentIteration
  -> Tensor           -- ^ trainLossTensor
  -> Tensor           -- ^ trainAccTensor
  -> Maybe Tensor     -- ^ validLossMaybe
  -> Maybe Tensor     -- ^ validAccMaybe
  -> Maybe FilePath   -- ^ newModelPath (if save)
  -> TrainingTracker
updateTracker tracker epochNum currentIteration trainLossTensor trainAccTensor validLossMaybe validAccMaybe mModelPath =
  tracker
    { currentEpoch   = epochNum
    , currentBatch   = currentIteration
    , trainLoss      = trainLoss tracker ++ [asValue trainLossTensor :: Float]
    , trainAccuracy  = trainAccuracy tracker ++ [asValue trainAccTensor :: Float]
    , validLoss      = maybe (validLoss tracker) (\v -> validLoss tracker ++ [asValue v :: Float]) validLossMaybe
    , validAccuracy  = maybe (validAccuracy tracker) (\a -> validAccuracy tracker ++ [asValue a :: Float]) validAccMaybe
    , lastModelPath  = fromMaybe (lastModelPath tracker) mModelPath
    }




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

  let c = ModelConfig (nEmbd model) (nBlock model) (vocabSize model) (nHead model) (blockSize model) 
  let tracker0 = fromMaybe (initialTrainingTracker c) mTracker

  (finalModel, _validDl, finalTracker) <- foldM
    (\(curModel, curValidDl, curTracker) epoch -> do
       putStrLn $ "Starting epoch " ++ show epoch ++ "/" ++ show nbEpoch

       resetTrainDl <- if epoch > 1 then resetDataloader trainDataloader else pure trainDataloader
       let resetTracker = if epoch > 1 then curTracker { currentBatch = 0 } else curTracker
       processEpochLazy curModel resetTrainDl curValidDl optimizer nbEpoch epoch resetTracker
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





