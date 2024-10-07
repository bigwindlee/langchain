from langchain_openai import AzureOpenAI
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

llm = AzureOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0,
    api_version="2023-03-15-preview"
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    deployment_name="text-embedding-3-large",
    api_version="2023-05-15"
)

Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(
    input_files=["./data/paul_graham/paul_graham_essay.txt"]
).load_data()

index = VectorStoreIndex.from_documents(documents)

# 测试查询
query_engine = index.as_query_engine()
question = "What are the key points discussed in the essay?"
response = query_engine.query(question)
print(response)
