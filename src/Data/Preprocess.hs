module Data.Preprocess where

import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Word (Word8)
import Data.Char (toLower)
import Codec.Binary.UTF8.String (encode)
import qualified Data.Map.Strict as M

allowedChars :: [Char]
allowedChars = "abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüæœ "

isAllowedChar :: Word8 -> Bool
isAllowedChar w =
  let c = head (encode [toLower (toEnum (fromEnum w))])
  in c `elem` (map (head . encode . (:[])) allowedChars)

preprocess ::
  B.ByteString -> -- input
  [B.ByteString]  -- wordlist per line
preprocess texts =
  let cleaned  = B.pack $ filter isAllowedChar (B.unpack texts)
      lowered  = BL.map toLower cleaned
  in  B.split (head $ encode " ") lowered


wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0..]))