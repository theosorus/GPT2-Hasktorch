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

getModelPath :: String -> String -> Int -> Int -> FilePath
getModelPath modelName modelDir nEpoch nBatch = 
    modelDir ++ "/" ++ modelName ++ "_e" ++ show nEpoch ++ "_b" ++ show nBatch ++ ".pt"


loadModel :: FilePath -> ModelConfig ->IO Model 
loadModel path config = do
    model <- modelInit config
    loadedModel <- loadParams model path
    pure loadedModel







