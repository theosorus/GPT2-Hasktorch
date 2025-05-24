module Model.Save where 

import Torch (saveParams,loadParams)

import Model.GPT

saveModel :: FilePath -> Model -> Bool -> IO ()
saveModel path model verbose = do
    saveParams model path
    if verbose then
        putStrLn $ "Model saved at: " ++ path
    else
        pure ()


loadModel :: FilePath -> ModelConfig ->IO Model 
loadModel path config = do
    model <- modelInit config
    loadParams model path
    pure model







