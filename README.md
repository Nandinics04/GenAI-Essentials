# GenAI Essentials

Personal learning repo for applied generative AI. The goal is not to train a foundation model. It is to build the pieces an applied AI engineer actually ships: load documents, split them, embed them, retrieve the right chunks, generate an answer from that context, and measure whether the answer is correct.

Everything lives in Jupyter notebooks. Run each notebook from the top with the `.venv` kernel. Notebooks sit in `notebook/`, so project-root files are reached with `../` (`../data/pdf`, `../text.txt`, `../books.jsonl`). `load_dotenv()` walks up from the notebook and finds `.env` at the repo root.

Python 3.13. Package manager is [uv](https://docs.astral.sh/uv/) 0.12. Chat goes through OpenRouter. Embeddings run locally with `all-MiniLM-L6-v2`.

## What this repo covers

1. Pydantic models for validated, structured data.
2. LangChain documents, text and PDF loaders, and chunking.
3. Local sentence-transformer embeddings and a persistent Chroma store.
4. A retriever that turns distance into a similarity score, then a RAG prompt.
5. The same idea on two other stores: Typesense (keyword search) and MongoDB Atlas (vector search).
6. An agent that decides whether to retrieve, built as a LangGraph state machine.
7. LangSmith datasets and evaluators for a plain chatbot and for a RAG bot.

## Setup

From the project root in PowerShell:

```powershell
uv venv
uv sync
uv pip install -r requirements.txt
.venv\Scripts\Activate.ps1
```

`uv sync` installs the locked dependencies in `pyproject.toml` and `uv.lock`. `uv pip install -r requirements.txt` adds packages listed only in requirements. `uv add` is the command that edits `pyproject.toml`. `uv pip install` does not.

Leave the virtual environment with `deactivate`. Close the terminal with `exit`.

If activation fails with `UnauthorizedAccess` on `Activate.ps1`, PowerShell is blocking scripts. Allow them for the current user, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Select the `.venv` kernel in Jupyter. After you change a class in a cell, rerun that cell. The kernel keeps the old class until you do.

### Environment variables

Create `.env` in the project root. It is gitignored. Never commit it, and never paste a key into a notebook cell.

```text
OPENROUTER_API_KEY=
LANGSMITH_API_KEY=
GROQ_API_KEY=
```

| Variable | Where it is used |
| --- | --- |
| `OPENROUTER_API_KEY` | Chat and the LangSmith grader. The OpenAI SDK is pointed at `https://openrouter.ai/api/v1` |
| `LANGSMITH_API_KEY` | Tracing and experiments. Also set `LANGSMITH_TRACING=true` |
| `GROQ_API_KEY` | The first chat experiments in `document.ipynb`. Later notebooks moved chat to OpenRouter |

Typesense and MongoDB credentials belong in `.env` or the provider console, not in source.

## How the pieces fit

A RAG app in this repo always does the same four steps:

1. **Load.** A file or web page becomes one or more LangChain `Document` objects. Each document has `page_content` and `metadata`.
2. **Split.** A long document is cut into chunks (about 1000 characters, 100–200 overlap) so retrieval can return a relevant passage instead of the whole file.
3. **Embed and store.** Each chunk is turned into a vector with `all-MiniLM-L6-v2` and saved in Chroma, FAISS, Typesense, an in-memory store, or MongoDB Atlas.
4. **Retrieve, then generate.** The question is embedded the same way. The nearest chunks become the prompt context. The chat model answers from that context. It does not answer from the vector store by itself. Retrieval returns text. The LLM writes the answer.

Citations in the advanced pipeline are labels, not a separate knowledge source. Each hit keeps the source file, page, similarity score, and a short preview so you can see which chunk the answer came from.

## 1. Pydantic — `pydantic/intro.ipynb`

Pydantic models are Python classes that check data at runtime. Later notebooks use the same idea (`TypedDict` grade schemas) so an LLM judge must return a fixed shape instead of free text.

What the notebook builds:

- A `BaseModel` compared with a plain `@dataclass`. The dataclass only stores fields. The Pydantic model rejects a wrong type.
- `Optional` fields. `Optional[str] = None` means the field may be missing. Required fields still have to be passed.
- Nested models. `Customer` contains an `Address`. You can pass the address as a dict and Pydantic builds the inner model.
- `Field` constraints. `Field(gt=0, le=1000)` rejects a price of 1500 with a `ValidationError`.
- Coercion that does work: a numeric string can become an `int`, and a tuple can become a `list`. A string where an `int` is required inside a list does not.

## 2. Documents, embeddings, Chroma, and RAG — `notebook/document.ipynb`

This is the main RAG notebook. It goes from a single `Document` to a pipeline that retrieves PDF chunks and answers with an LLM.

### Documents and loaders

- `Document(page_content=..., metadata=...)` is the unit everything else uses. Metadata here includes source, page, and author.
- `TextLoader` reads one file, for example `../data/text_files/python_intro.txt`. One file becomes one document until you split it.
- `DirectoryLoader` reads every `*.txt` under a folder.
- PDFs under `data/pdf` are loaded with PyMuPDF and a directory loader, then split.

Sample files used while learning: `text.txt`, `sample.txt`, `calculus.txt`, `linear_algebra.txt`, `data/text_files/python_intro.txt`, `data/text_files/machine_learning.txt`, and the PDFs in `data/pdf`.

### Chunking

`RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`. Overlap keeps a sentence that falls on a boundary inside two neighboring chunks.

`CharacterTextSplitter` (used later with Typesense) splits on `\n\n` by default. If you pass the original documents into the vector store instead of the chunks, the store holds one record: the whole file. The splitter does nothing until you store `chunks`.

On the PDF set this produced 173 unique chunks. The Chroma collection `pdf_documents` later showed 346 rows because the same chunks were inserted twice. A second add brought the count to 519. Re-running ingestion appends. It does not replace.

### Embeddings

`EmbeddingManager` wraps `SentenceTransformer("all-MiniLM-L6-v2")`.

- `__init__` stores `self.model_name` and calls `_load_model()`, which sets `self.model`. Other methods use `self.model`, not the constructor argument directly.
- `generate_embeddings(texts)` returns a NumPy array of shape `(len(texts), 384)`.
- The model is local and free. The first run downloads it. `OpenAIEmbeddings` calls a paid API, which is why the later notebooks switched off it.
- If a notebook says `EmbeddingManager has no attribute generate_embeddings` after you rename the method, the kernel still has the old class. Rerun the class cell and create a new `embedding_manager`.

### Chroma

`VectorStore` opens a persistent client at `../data/vector_store` and a collection named `pdf_documents`.

That folder is gitignored. It was committed once; remove it from the index without deleting the local files with:

```powershell
git rm -r --cached data/vector_store
```

### Retrieval

`RAGRetriever.retrieve(query, top_k, score_threshold)`:

1. Embeds the query with the same model used at index time.
2. Calls `collection.query`.
3. Chroma returns L2 distance, not a similarity. For these queries the distance was about `1.1`.
4. `1 - distance` is negative, so a threshold of `0` or `0.3` drops every hit and prints "No documents found".
5. The conversion that matches this distance is `similarity = 1 - (distance ** 2) / 2`, which is about `0.39` for a distance of `1.1`. The threshold used after that is `0.3`.
6. The returned dict key is `document`, not `content`.

`retrieve` returns chunks. It does not call an LLM. An empty result means the score filter removed the hits, or the query embedding does not match what was stored.

### Generation

`rag_simple` joins the retrieved `document` fields into a context string and sends one prompt to the chat model. The prompt is already an f-string. Calling `.format()` on it again breaks if the context contains `{` or `}`.

`rag_advanced` adds a minimum score and returns the context when asked. `AdvancedRAGPipeline` adds citations (source, page, score, preview), a short history, optional streaming, and optional summarization.

In Jupyter, `answer = rag_simple(...)` does not display the answer. The last line of the cell has to be `answer`, or you print it.

### Chat model changes in this notebook

- First model was Groq `gemma2-9b-it`. Groq decommissioned it. The replacement used here is `openai/gpt-oss-20b` on Groq (`llama-3.1-8b-instant` was the intermediate option and is itself scheduled to shut down).
- Later notebooks stopped using Groq for chat and call OpenRouter instead.

### Bugs this notebook cleared

| What broke | Why | Fix |
| --- | --- | --- |
| `UnboundLocalError` on `context` | `context:` was a type annotation, not an assignment | `context = "..."` |
| `KeyError: 'content'` | Retriever dict key is `document` | Read `doc['document']` |
| No documents retrieved | Similarity was `1 - distance` on an L2 distance around 1.1 | `1 - (distance ** 2) / 2` and threshold `0.3` |
| Only the first hit returned | `return` sat inside the loop | Return after the loop |
| `AttributeError` on the vector store | Method was `_initialze_store` and the client was `self.clint` | `_initialize_store` and `self.client` |
| Groq `model_decommissioned` | `gemma2-9b-it` was removed | `openai/gpt-oss-20b` |
| Import errors on newer LangChain | `langchain.schema` and `langchain.text_splitter` moved | `langchain_core.documents.Document` and `langchain_text_splitters` |
| `HuggingFaceEmbeddings` import error | It left `langchain.embeddings` | `uv pip install langchain-huggingface`, then `from langchain_huggingface import HuggingFaceEmbeddings` |

## 3. Typesense — `notebook/typesense.ipynb`

Typesense is a search engine. This notebook uses it two ways: a books collection queried by keywords, then a LangChain vector store for RAG.

- Client talks to a Typesense Cloud host on port 443 over HTTPS.
- `books.jsonl` in the project root is the book catalog. From the notebook the path is `../books.jsonl`.
- Search example: `query_by` title and authors, `sort_by` `ratings_count:desc`.
- The RAG half loads `../text.txt` with `TextLoader`, splits with `CharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=100`), embeds with `HuggingFaceEmbeddings`, and indexes with `Typesense.from_documents`.
- The collection parameter is `typesense_collection_name` (underscore). A hyphen raises `TypeError`.
- `typesense_api_key` must be an admin key. A search-only key returns 401.
- `from_documents` must receive `chunks`. Passing the unsplit `documents` stores a single record, which looked like "chunking did nothing".
- To reindex, delete or recreate the collection first. A second insert does not replace the old document.

## 4. MongoDB Atlas vector search — `notebook/rag.ipynb`

Same RAG shape on Atlas: embed text, insert into `sample_mflix.ragpdf`, then `aggregate` with a vector search index, then send the hits to a chat model.

What had to be true before the query worked:

- The URI user is a Database Access user, not the Atlas website login.
- Characters in the password must be percent-encoded in the URI. `#` becomes `%23`. `&` becomes `%26`. An auth failure (`bad auth`, code 8000) means the user or password does not match Database Access. Encoding does not fix a wrong password.
- Call `collection.aggregate(...)`. The collection is already `ragpdf`, so `collection.ragpdf.aggregate` is wrong.
- Do not name the Mongo client `client` if an OpenAI client already uses that name. The embedding function then calls `embeddings.create` on a collection. The notebook uses `openai_client` and `mongo_client`.
- `embeddings.create` takes `input=text`, not `text=`. The model id needs a provider prefix, such as `openai/text-embedding-3-small`. That embedding model is paid. The chat side of this notebook uses OpenRouter.
- TLS errors on the shard hosts were fixed with `certifi`: `MongoClient(uri, tlsCAFile=certifi.where())` after `uv pip install certifi`.
- Atlas Network Access must allow the current public IP. The IP changes between networks.
- A DNS timeout on `mongodb+srv` (`LifetimeTimeout` against the router at `192.168.1.1`) was worked around by pointing dnspython at public resolvers before creating the client:

```python
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
```

## 5. Agentic RAG — `notebook/1-agenticrag.ipynb`

A LangGraph workflow that sometimes retrieves and sometimes answers directly.

State is a `TypedDict` named `AgentState`: `question`, `documents`, `answer`, `need_retrieval`.

Four sample sentences about LangGraph, RAG, vector databases, and agents are split only by being stored as separate `Document`s, embedded with `HuggingFaceEmbeddings`, and indexed in FAISS. `vectorstore.as_retriever(k=3)` is the retriever.

Nodes:

| Node | What it does |
| --- | --- |
| `decide` | Sets `need_retrieval` if the question contains what, how, explain, describe, or tell me |
| `retrieve` | `retriever.invoke(question)` and writes `documents` |
| `generate` | If documents exist, prompts with that context. Otherwise asks the model directly |

Edges: entry is `decide`. A conditional edge goes to `retrieve` or straight to `generate`. `retrieve` always continues to `generate`. `generate` goes to `END`.

`ask_question` builds the initial state and calls `app.invoke`. The chat model is OpenRouter through `ChatOpenAI` (`model_provider` style: OpenAI client, `base_url` `https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`).

This graph is a fixed heuristic, not a planner. The keyword check is the policy. A question without those words skips retrieval.

## 6. Evaluation — `notebook/1-rag_evaluation.ipynb`

Two experiments. Both are traced in LangSmith under the dataset name `Simple Evaluation Chatbot` for the first half, then a separate RAG dataset for the blog-post bot.

### Chatbot experiment

Five question and reference-answer rows (LangSmith, OpenAI, Google, Mistral, and similar). `my_app` calls OpenRouter and returns the message text. The LangSmith target must return a dict:

```python
def ls_target(inputs):
    return {"response": my_app(inputs["question"])}
```

Evaluators:

- **Correctness.** A second model call grades the prediction against `reference_outputs['answer']` and must reply `CORRECT` or `INCORRECT`. The function is named `corectness` in the notebook.
- **Concision.** `len(outputs['response']) < 2 * len(reference_outputs['answer'])`.

`evaluate(...)` runs the target on every example and then the evaluators. `experiment_prefix` is `Evaluation Chatbot`. `experimental_results.to_pandas()` needs `pandas` (`uv pip install pandas`).

If the target raises, LangSmith still runs the evaluators. They read `outputs['response']` and raise `KeyError: 'response'` because the dict was never returned. The `KeyError` is a consequence. The first traceback is the real failure.

Dataset creation returns 409 if `Simple Evaluation Chatbot` already exists. Read it instead of creating it again:

```python
client.read_dataset(dataset_name="Simple Evaluation Chatbot")
```

### RAG experiment

Loads three posts with `WebBaseLoader`:

- LLM Powered Autonomous Agents
- Prompt Engineering
- Adversarial Attacks on LLMs

`WebBaseLoader` imports BeautifulSoup (`bs4`). Install it with `uv pip install beautifulsoup4` if that import fails.

Chunks use `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (`chunk_size=1000`, `chunk_overlap=200`) and go into `InMemoryVectorStore`. Embeddings are `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`, not `OpenAIEmbeddings()`, because the OpenAI embedding API was rate-limiting.

`rag_bot` is `@traceable`. It retrieves, stuffs the chunks into the system prompt, and asks the chat model to answer in three sentences or say it does not know. The return value is `{"answer": ..., "documents": docs}`.

The chat model is:

```python
init_chat_model(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0,
)
```

`model_provider="openai"` means "speak the OpenAI HTTP API". The request still goes to OpenRouter. The model id is the OpenRouter id.

RAG evaluators, each an LLM judge with a schema:

| Metric | Compares | Passes when |
| --- | --- | --- |
| Correctness | Response vs reference answer | The facts match the reference |
| Relevance | Response vs the user question | The response actually answers the question |
| Groundedness | Response vs retrieved documents | Claims are supported by the retrieved text |
| Retrieval relevance | Retrieved documents vs the question | The chunks are about the question |

Grade schemas (`CorrectnessGrade`, `RelevanceGrade`, `GroundedGrade`, `RetrievalRelevanceGrade`) force the judge to return a boolean explanation structure instead of a paragraph.

## OpenRouter, and why models kept failing

OpenRouter is not OpenAI. The SDK shape is the same:

```python
OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
```

LangSmith tracing wraps that client with `wrappers.wrap_openai(...)`.

`openrouter/free` is a router. It picks a free model at random, which makes evaluation runs hard to compare. A fixed id is better.

Free Google models (`google/gemma-4-31b-it:free` and `google/gemma-4-26b-a4b-it:free`) both returned HTTP 429 from Google AI Studio's shared pool (`limit_source: upstream_provider_shared_pool`). Switching from one Gemma model to the other does not help. The model used after that is `nvidia/nemotron-3-super-120b-a12b:free`, in both `my_app` and the grader.

Embeddings stay on Hugging Face. `openai/text-embedding-3-small` through OpenRouter is a paid embedding model.

## Repository layout

```text
GenAI-Essentials/
├── pydantic/intro.ipynb          # validation
├── notebook/
│   ├── document.ipynb            # load, chunk, Chroma, RAG
│   ├── typesense.ipynb           # keyword search and Typesense RAG
│   ├── rag.ipynb                 # MongoDB Atlas vector search
│   ├── 1-agenticrag.ipynb        # LangGraph retrieve-or-answer graph
│   └── 1-rag_evaluation.ipynb    # LangSmith chatbot and RAG evals
├── data/
│   ├── pdf/                      # source PDFs
│   ├── text_files/               # short text corpora
│   └── vector_store/             # local Chroma data, gitignored
├── books.jsonl                   # Typesense book catalog
├── text.txt                      # sample document for loaders
├── src/genai/                    # package stub; `genai` prints a hello line
├── pyproject.toml
├── requirements.txt
└── .env                          # local only
```

## Practices this project settled on

- Keys live in `.env`. A Groq key pasted into `notebook/typesense.ipynb` was blocked by GitHub push protection. Remove the key from the notebook, and rotate any key that was committed or pasted.
- `.gitignore` does not untrack files that are already committed. Use `git rm -r --cached` for `data/vector_store/`.
- After editing a class, rerun the cell and rebuild the object. The kernel will not pick up the new method on the old instance.
- Retrieval code and generation code fail differently. "No documents found" is the retriever. A printed retrieval with no answer is the notebook not displaying the return value. A 429 is the chat provider. A `KeyError: 'response'` during eval means the target crashed before it returned that key.
- Keep one embedding model for indexing and querying. Mixing OpenAI embeddings at query time with MiniLM vectors in the index will not retrieve anything useful.
