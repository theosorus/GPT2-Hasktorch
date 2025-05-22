# GPT2 Hasktorch implementation

The goal of this project is to reproduce GPT-2, created by OpenAI, in the Haskell programming language using the Hasktorch library, drawing inspiration from Andrej Karpathy's implementation in PyTorch.

**Haskell** : https://www.haskell.org/

**Haskorch** : http://hasktorch.org/

**Nano GPT(Karpathy's implementation)** : https://github.com/karpathy/nanoGPT


## Launch the program

```bash
docker compose up -d
```

```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/GPT2/ && stack run"
```

```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/GPT2/ && stack test"
```

## use Jupyter
```http://localhost:8890/lab```
