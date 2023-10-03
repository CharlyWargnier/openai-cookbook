import concurrent
import os
import subprocess
from ast import literal_eval
from typing import Iterator, List

import numpy as np
import openai
import pandas as pd
import streamlit as st
import tiktoken
from langchain.chains import RetrievalQA
# Langchain imports
from langchain.embeddings import OpenAIEmbeddings
from numpy import array, average
from redis import Redis as r
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# Setup Redis

REDIS_HOST = "localhost"
REDIS_PORT = "6379"
REDIS_DB = "0"

redis_client = r(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

# if can't ping redis, then start redis
try:
    redis_client.ping()
except:
    subprocess.Popen("redis-stack-server")
    import time

    time.sleep(5)
    redis_client = r(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False
    )


@st.cache_data
def load_article():
    return pd.read_csv("./wikipedia_articles_2000.csv")


article_df = load_article()

# Constants
VECTOR_DIM = 1536  # length of the vectors
PREFIX = "wiki"  # prefix for the document keys
DISTANCE_METRIC = "COSINE"  # distance metric for the vectors (ex. COSINE, IP, L2)

# Create search index

# Index
INDEX_NAME = "wiki-index"  # name of the search index
VECTOR_FIELD_NAME = "content_vector"

# Define RediSearch fields for each of the columns in the dataset
# This is where you should add any additional metadata you want to capture
id = TextField("id")
url = TextField("url")
title = TextField("title")
text_chunk = TextField("content")
file_chunk_index = NumericField("file_chunk_index")

# define RediSearch vector fields to use HNSW index

text_embedding = VectorField(
    VECTOR_FIELD_NAME,
    "HNSW",
    {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": DISTANCE_METRIC},
)
# Add all our field objects to a list to be created as an index
fields = [url, title, text_chunk, file_chunk_index, text_embedding]

# Check if index exists
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    # Create RediSearch Index
    print("Not there yet. Creating")
    redis_client.ft(INDEX_NAME).create_index(
        fields=fields,
        definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
    )


# We'll use 1000 token chunks with some intelligence to not split at the end of a sentence
TEXT_EMBEDDING_CHUNK_SIZE = 1000
EMBEDDINGS_MODEL = "text-embedding-ada-002"

## Chunking Logic

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def get_unique_id_for_file_chunk(title, chunk_index):
    return str(title + "-!" + str(chunk_index))


def chunk_text(x, text_list):
    url = x["url"]
    title = x["title"]
    file_body_string = x["text"]

    """Return a list of tuples (text_chunk, embedding) for a text."""
    token_chunks = list(chunks(file_body_string, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [
        f"Title: {title};\n" + tokenizer.decode(chunk) for chunk in token_chunks
    ]

    # Get the vectors array of triples: file_chunk_id, embedding, metadata for each embedding
    # Metadata is a dict with keys: filename, file_chunk_index

    for i, text_chunk in enumerate(text_chunks):
        id = get_unique_id_for_file_chunk(title, i)
        text_list.append(
            (
                {
                    "id": id,
                    "metadata": {
                        "url": x["url"],
                        "title": title,
                        "content": text_chunk,
                        "file_chunk_index": i,
                    },
                }
            )
        )


## Batch Embedding Logic

# Simple function to take in a list of text objects and return them as a list of embeddings
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input: List):
    response = openai.Embedding.create(
        input=input,
        model=EMBEDDINGS_MODEL,
    )["data"]
    return [data["embedding"] for data in response]


def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


# Function for batching and parallel processing the embeddings
def embed_corpus(
    corpus: List[str],
    batch_size=64,
    num_workers=8,
    max_context_len=8191,
):

    # Encode the corpus, truncating to max_context_len
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_corpus = [
        encoded_article[:max_context_len]
        for encoded_article in encoding.encode_batch(corpus)
    ]

    # Calculate corpus statistics: the number of inputs, the total number of tokens, and the estimated cost to embed
    num_tokens = sum(len(article) for article in encoded_corpus)
    cost_to_embed_tokens = num_tokens / 1_000 * 0.0004
    print(
        f"num_articles={len(encoded_corpus)}, num_tokens={num_tokens}, est_embedding_cost={cost_to_embed_tokens:.2f} USD"
    )

    # Embed the corpus
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = [
            executor.submit(get_embeddings, text_batch)
            for text_batch in batchify(encoded_corpus, batch_size)
        ]

        with tqdm(total=len(encoded_corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(batch_size)

        embeddings = []
        for future in futures:
            data = future.result()
            embeddings.extend(data)

        return embeddings


if int(redis_client.ft(INDEX_NAME).info()['num_docs']) < 1:
    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # List to hold vectors
    text_list = []

    # Process each PDF file and prepare for embedding
    x = article_df.apply(lambda x: chunk_text(x, text_list),axis = 1)

    # Batch embed our chunked text - this will cost you about $0.50
    embeddings = embed_corpus([text["metadata"]['content'] for text in text_list])

    # Join up embeddings with our original list
    embeddings_list = [{"embedding": v} for v in embeddings]
    for i,x in enumerate(embeddings_list):
        text_list[i].update(x)

    # Create a Redis pipeline to load all the vectors and their metadata
    def load_vectors(client:r, input_list, vector_field_name):
        p = client.pipeline(transaction=False)
        for text in input_list:    
            #hash key
            key=f"{PREFIX}:{text['id']}"
            
            #hash values
            item_metadata = text['metadata']
            #
            item_keywords_vector = np.array(text['embedding'],dtype= 'float32').tobytes()
            item_metadata[vector_field_name]=item_keywords_vector
            
            # HSET
            p.hset(key,mapping=item_metadata)
                
        p.execute()

    batch_size = 100  # how many vectors we insert at once

    for i in range(0, len(text_list), batch_size):
        # find end of batch
        i_end = min(len(text_list), i+batch_size)
        meta_batch = text_list[i:i_end]
        
        load_vectors(redis_client,meta_batch,vector_field_name=VECTOR_FIELD_NAME)