# One-time setup 

## 1. Install BEIR within the environment

```pip install beir```

## 2. Install Elasticsearch on the Ubuntu server:

Please follow the instructions carefully for the installation (`Configuring Elasticsearch` is not needed)

https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/#installing-elasticsearch


## 3. Search engine initialization
Run ```python bm25_init.py``` to load the config and re-indexing 

This may take ~10 minutes.

# Usage

```bm25_search(query)``` function from ```bm25_search.py``` supports BM25 search on any text query within HotPotQA knowledge base.

See example usage within ```bm25_search.py```
