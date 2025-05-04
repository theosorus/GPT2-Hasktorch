module Utils where


import Torch
import Torch.Functional as F
import Torch.DType




randInt :: [Int] -> Int -> Int -> IO Tensor
randInt dims min max = do
    let size = map fromIntegral dims
    randomTensor <- randIO' size
    let scaledTensor = F.mul (F.add randomTensor (asTensor (-min))) (asTensor (max - min))
    return $ F.toDType Int64 (F.floor scaledTensor)