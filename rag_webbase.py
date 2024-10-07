"""
Tutorials > Build a Retrieval Augmented Generation (RAG) App
https://python.langchain.com/docs/tutorials/rag/

LOGS:
    2024-10-07 worked.
"""
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = AzureOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0,
    api_version="2023-03-15-preview"
)

# You need to deploy your own embedding model as well as your own chat completion model
# 坑：
#   AzureOpenAIEmbeddings需要的环境变量是：AZURE_OPENAI_API_KEY
#   而可以从OPENAI_API_KEY中获得api_key
embed_model = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large",
    openai_api_version="2023-05-15"
)

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embed_model
)

# Retrieve and generate using the relevant snippets of the blog.
# <class 'langchain_core.vectorstores.base.VectorStoreRetriever'>
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

# RunnableSequence（即执行链）通过重载 | 运算符，将不同的步骤或组件串联起来，形成一个流水线。
# 因此，虽然字典本身没有重载 | 运算符，但参与 | 运算符的另一个对象（这里是 prompt）
# 是 Runnable 类的一个实例，该类实现了 __or__ 方法。
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")
print(response)
