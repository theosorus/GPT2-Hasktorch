module Training where

import Torch
import Model.GPT
import Data.Dataloader (DataLoader, Batch)




-- trainBatch :: (Optimizer opt) => Model -> Batch -> opt -> Double -> IO (Model, Tensor, Tensor)
-- trainBatch model batch optimizer lr = do
--     let (output,loss) = processBatch model batch
--     (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
--     pure (newModel,loss,output)
    
    