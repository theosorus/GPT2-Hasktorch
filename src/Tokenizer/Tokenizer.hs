{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Tokenizer.Tokenizer (tokenMain) where 

import GHC.Generics (Generic)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as BL
import qualified Data.ByteString.UTF8 as BSU
import Data.Aeson (decode)
import Data.Word
import qualified Data.Map as Map
import Data.Maybe (fromMaybe)
import System.IO
import Data.List.Split (splitOn)
import Distribution.Simple
import qualified Data.Text as T
import Data.Word (Word8)

type CharMap = Map.Map String Int

data Tokenizer = Tokenizer
    {
        fullVocab :: CharMap,
        pairs :: [(String, String)],
        inputText :: String
    } deriving (Generic)
    
loadJSON :: FilePath -> IO (Maybe CharMap)
loadJSON filePath = do
  content <- B.readFile filePath
  return $ decode content
    
readFileToPairs :: FilePath -> String -> IO [(String, String)]
readFileToPairs filePath delimiter = do
    content <- readFile filePath
    let lines' = lines content         -- Diviser en lignes
        pairs = map (toPair delimiter) lines'   -- Convertir chaque ligne en tuple
    return pairs
  where
    -- Fonction pour convertir une ligne en tuple (s1, s2)
    toPair :: String -> String -> (String, String)
    toPair delim line = 
        case splitOn delim line of
            [first, second] -> (first, second)  -- Cas normal: deux parties


getVocab :: Maybe CharMap -> IO CharMap
getVocab maybeVocab =
    case maybeVocab of
        Just vocab -> return vocab
        Nothing    -> do
            putStrLn "Erreur: vocab non chargé"
            return Map.empty

toByte :: BS.ByteString -> IO String
toByte bs = do
    let toW8 = BS.unpack bs
    return (BSU.toString (BS.pack toW8))

initializeTokenizer :: BS.ByteString -> FilePath -> FilePath -> IO Tokenizer
initializeTokenizer bsVal vocabPath mergePath = do
    maybeVocab <- loadJSON vocabPath
    vocab <- getVocab maybeVocab
    mergePairs <- readFileToPairs mergePath " "
    bs <- toByte bsVal
    return Tokenizer
        { 
          fullVocab = vocab,
          pairs = mergePairs,
          inputText = bs
        }


-- Replace space by "Ġ" to match GPT2 datas (vocab.json & merges.txt)
replaceSpace :: Tokenizer -> String
replaceSpace tokenizer = 
    T.unpack (T.replace (T.pack " ") (T.pack "Ġ") (T.pack (inputText tokenizer)))

makeStrArray :: String -> [String]
makeStrArray =
    map (:[])

merges :: Tokenizer -> [String] -> [String]
merges tokenizer tokens = 
    case pairs tokenizer of
        [] -> tokens
        (cp:rp) ->
            let newTokenizer = tokenizer { pairs = rp }
                mergedTokens = merge cp tokens
            in merges newTokenizer mergedTokens
    
merge :: (String, String) -> [String] -> [String]
merge _ [] = []
merge _ [x] = [x]
merge (a, b) (x1:x2:xs)
    | x1 == a && x2 == b = (a ++ b) : merge (a, b) xs
    | otherwise          = x1 : merge (a, b) (x2:xs)

changeToIndex :: Tokenizer -> [String] -> [Int]
changeToIndex tokenizer tokens =
    map (\c -> Map.findWithDefault 50257 c (fullVocab tokenizer)) tokens

untokenizer :: [Int] -> CharMap -> String
untokenizer tokensId strKey =
    let idsKey = reverseMap strKey
        find = concat (map (\i -> Map.findWithDefault "?" i idsKey) tokensId)
        result = T.unpack (T.replace (T.pack "Ġ") (T.pack " ") (T.pack find))
    in result

reverseMap :: CharMap -> Map.Map Int String
reverseMap vocabMap =
    let vocab = Map.toList vocabMap
        reverseVocab = map (\(a, b) -> (b, a)) vocab
    in Map.fromList reverseVocab

tokenMain ::  BS.ByteString -> FilePath -> FilePath -> IO [Int]
tokenMain input vocabPath pairsPath = do
    tokenizer <- initializeTokenizer input vocabPath pairsPath
    let validArrayString = makeStrArray (replaceSpace tokenizer)  -- Makes a valid array of String as ["h","e","l","l","o","Ġ","!"]
        merged = merges tokenizer validArrayString -- Merged following GPT2 merges.txt
    return (changeToIndex tokenizer merged)

