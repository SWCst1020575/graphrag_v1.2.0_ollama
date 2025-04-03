# Use Ollama to integrate into GraphRAG
## Install from source
```sh
git clone https://github.com/SWCst1020575/graphrag_ollama_v1.2.0.git
cd graphrag_ollama_v1.2.0
pip install -e .
pip install ollama
```


## Init GraphRAG 
```sh
graphrag init --root ragtest
```
Please modify **settings.yaml**.

**Must modify**
1. llm: **model**, **api_base**
2. embeddings: **model**, **api_base**
3. chunks: If using Chinese, the size should be set to 300 or below; it won't run if it's too large (providing too short of an input also can't run).


## Run GraphRAG 
```sh
graphrag index --root ragtest
```

## Query
```sh
graphrag query --root ragtest --method local --query "query ..."
graphrag query --root ragtest --method global --query "query ..."
```