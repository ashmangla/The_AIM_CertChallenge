"""
RAG Evaluation using RAGAS with Synthetic Data Generation (SDG)

This script evaluates the RAG system using RAGAS metrics:
- Context Recall: How well does the retrieval capture relevant information?
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: How relevant is the answer to the question?
- Context Entity Recall: Entity-level recall
- Factual Correctness: Factual accuracy of the response

We test 3 retrieval strategies:
1. Baseline (Naive): Simple vector similarity search
2. Cohere Rerank: Use Cohere's reranking model for better relevance
3. Multi-Query: Generate multiple queries for better coverage

The best performing strategy will be saved to a config file for production use.
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from operator import itemgetter

# Add the parent directory to the path to import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION - Load API keys from environment variables
# ============================================================================
# For local development: Copy api/.env.example to api/.env and fill in your keys
# For production: Set these as environment variables in your deployment platform

# Load from .env file if it exists (for local development)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Validate that keys are set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file or environment variables.")
if not COHERE_API_KEY:
    print("⚠️  WARNING: COHERE_API_KEY not set. Cohere Rerank evaluation will be skipped.")

# Set API keys in environment (for LangChain/RAGAS modules)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Enable RAGAS debug tracking for better visibility
os.environ["RAGAS_DEBUG_TRACKING"] = "True"
os.environ["RAGAS_DO_NOT_TRACK"] = "false"  # Enable tracking for debugging
os.environ["RAGAS_DEBUG"] = "1"  # Enable debug mode
os.environ["LANGCHAIN_VERBOSE"] = "true"  # Enable LangChain verbose mode

# Set logging level for more visibility
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("ragas").setLevel(logging.DEBUG)

print(f"✓ API keys configured")
print(f"✓ RAGAS debug tracking enabled")
print(f"✓ Verbose logging enabled")
# ============================================================================

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAGAS imports (0.3.7 compatible)
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
    AspectCritic
)
from ragas.testset import TestsetGenerator

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
GOLDEN_DATASET_DIR = Path(__file__).parent / "golden_dataset"
RESULTS_DIR.mkdir(exist_ok=True)
GOLDEN_DATASET_DIR.mkdir(exist_ok=True)

# RAG Prompt Template
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""


def load_and_prepare_documents(language_filter='en', chunking_strategy='recursive', embeddings=None):
    """Load and prepare documents for evaluation with language filtering and configurable chunking.
    
    Args:
        language_filter: Language code to filter by (default: 'en' for English).
                        Set to None to include all languages.
        chunking_strategy: 'recursive' for RecursiveCharacterTextSplitter or 
                          'semantic' for SemanticChunker (default: 'recursive')
        embeddings: Required for semantic chunking
    """
    print("📚 Loading documents...")
    
    # Load documents
    loader = DirectoryLoader(str(DATA_DIR), glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document pages")
    
    # Apply language filter if specified
    if language_filter:
        print(f"🌐 Filtering for language: {language_filter.upper()}")
        from langdetect import detect, LangDetectException
        
        filtered_docs = []
        for doc in docs:
            try:
                # Detect language of the page content
                detected_lang = detect(doc.page_content[:500])  # Use first 500 chars for detection
                if detected_lang == language_filter:
                    filtered_docs.append(doc)
            except LangDetectException:
                # If detection fails, skip the document
                continue
        
        print(f"✅ Filtered to {len(filtered_docs)} {language_filter.upper()} pages (from {len(docs)} total)")
        docs = filtered_docs
    
    # Split into chunks based on strategy
    if chunking_strategy == 'semantic':
        if embeddings is None:
            raise ValueError("Embeddings required for semantic chunking")
        print(f"🔍 Using SemanticChunker (percentile breakpoint)...")
        text_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile"
        )
    else:  # recursive (default)
        print(f"📝 Using RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    rag_documents = text_splitter.split_documents(docs)
    print(f"✅ Split into {len(rag_documents)} {chunking_strategy} chunks")
    
    return docs, rag_documents


def create_vector_store(rag_documents, embeddings):
    """Create Qdrant vector store."""
    print("\n🔧 Creating vector store...")
    vectorstore = Qdrant.from_documents(
        rag_documents,
        embeddings,
        location=":memory:",
        collection_name="evaluation_chunks"
    )
    print(f"✅ Vector store created: {len(rag_documents)} documents")
    return vectorstore


def generate_testset_manual(docs, generator_llm, generator_embeddings):
    """Generate testset using manual KnowledgeGraph approach (for RAGAS 0.2+)."""
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer
    )
    
    print("\n📝 Generating testset using manual KnowledgeGraph approach...")
    
    # Check if KG already exists
    kg_path = GOLDEN_DATASET_DIR / "usecase_data_kg.json"
    
    if kg_path.exists():
        print(f"📂 Loading existing Knowledge Graph from {kg_path}")
        kg = KnowledgeGraph.load(str(kg_path))
        print(f"✅ Loaded Knowledge Graph with {len(kg.nodes)} nodes")
    else:
        print("🔧 Creating Knowledge Graph from documents...")
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
                )
            )
        print(f"✅ Created KG with {len(kg.nodes)} document nodes")
        
        print("🔄 Applying transforms to Knowledge Graph...")
        transforms = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)
        apply_transforms(kg, transforms)
        print("✅ Transforms applied")
        
        print(f"💾 Saving Knowledge Graph to {kg_path}")
        kg.save(str(kg_path))
    
    # Generate testset
    print("🎯 Generating test questions...")
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg)
    
    # Use only SingleHop queries for faster generation
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.0),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.0),
    ]
    
    testset = generator.generate(testset_size=5, query_distribution=query_distribution)
    test_set = testset.to_pandas()
    print(f"✅ Generated {len(test_set)} test questions")
    
    return test_set


def generate_or_load_testset(docs, generator_llm, generator_embeddings):
    """Generate new testset or load existing one with stratified sampling."""
    testset_path = GOLDEN_DATASET_DIR / "test_set.csv"
    
    if testset_path.exists():
        print(f"\n📂 Loading existing testset from {testset_path}")
        test_set = pd.read_csv(testset_path)
        print(f"✅ Loaded {len(test_set)} questions from existing testset")
    else:
        # Use RAGAS 0.2.10 API with parallel processing + STRATIFIED SAMPLING
        print("\n📝 Generating new testset using RAGAS 0.2.10 (with stratified sampling)...")
        
        # STRATIFIED SAMPLING: Sample ~80 documents for SDG (instead of all 120 pages)
        # This drastically speeds up testset generation while maintaining diversity
        import random
        random.seed(42)  # For reproducibility
        
        sample_size = min(80, len(docs))  # Use 80 pages max for SDG
        sampled_docs = random.sample(docs, sample_size)
        
        print(f"  📊 Using stratified sample: {sample_size} out of {len(docs)} documents")
        print(f"     (Full dataset will still be used for retrieval evaluation)")
        
        try:
            # Configure RunConfig for parallel execution
            print("  🔧 Configuring RunConfig with max_workers=4...")
            run_config = RunConfig(max_workers=4, max_wait=720)
            
            # Define homeowner persona for realistic question generation
            homeowner_persona = """You are generating questions from the perspective of a typical homeowner who:
- Is not a technical expert or appliance technician
- Wants quick, practical, actionable answers
- Asks straightforward "how-to" and troubleshooting questions
- Focuses on common tasks like maintenance, filter changes, and problem-solving
- Uses everyday, conversational language (not technical jargon)
- Is concerned about safety, cost, and ease of use

Generate questions that sound natural, such as:
- "How do I change the water filter in my fridge?"
- "What type of filter does this refrigerator use?"
- "Why is my ice maker not working?"
- "How often should I replace the air filter?"
- "Where can I buy replacement parts?"
- "What does this error code mean?"
- "How do I clean the condenser coils?"

Make questions specific to the appliance manual content, practical, and user-friendly."""
            
            # RAGAS 0.2.10 API with persona-guided LLM
            print("  🔧 Creating TestsetGenerator with homeowner persona...")
            generator_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7  # Slightly higher for more natural variation
            )
            
            generator = TestsetGenerator.from_langchain(
                llm=generator_llm,
                embedding_model=OpenAIEmbeddings()
            )
            
            # Inject persona into the generator's prompt (RAGAS 0.2.10 approach)
            # Note: We'll prepend the persona to each document for context
            persona_docs = []
            for doc in sampled_docs:
                # Add persona context to guide question generation
                doc_with_persona = doc.copy()
                doc_with_persona.metadata['user_persona'] = homeowner_persona
                persona_docs.append(doc_with_persona)
            
            print("  ✅ TestsetGenerator created successfully with homeowner persona")
            
            # Generate with run_config for parallel processing on SAMPLED docs
            print(f"  🚀 Generating {5} user-friendly test questions from {len(persona_docs)} sampled documents...")
            print(f"     - Using parallel processing with {4} workers")
            print(f"     - Questions will reflect typical homeowner language and concerns")
            dataset = generator.generate_with_langchain_docs(persona_docs, testset_size=5, run_config=run_config)
            print("  ✅ Test questions generated successfully")
            
            test_set = dataset.to_pandas()
            print(f"  ✅ Converted to pandas DataFrame: {len(test_set)} rows")
            
        except Exception as e:
            print(f"\n❌ Error during testset generation:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            import traceback
            print(f"   Traceback:\n{traceback.format_exc()}")
            raise
        
        # Alternative: Use manual approach (uncomment below and comment above for RAGAS 0.2+)
        # test_set = generate_testset_manual(docs, generator_llm, generator_embeddings)
        
        test_set.to_csv(testset_path, index=False)
        print(f"✅ Generated {len(test_set)} test questions")
        print(f"💾 Saved testset to {testset_path}")
    
    return test_set


def create_base_retriever(vectorstore):
    """Create base naive retriever to be reused by all methods."""
    print("\n🔧 Creating Base Retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    print("✅ Base retriever created")
    return retriever


def create_naive_retrieval_chain(naive_retriever, rag_prompt, chat_model):
    """Create baseline/naive retrieval chain."""
    print("\n🔧 Creating Naive Retrieval Chain...")
    
    chain = (
        {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    print("✅ Naive retrieval chain created")
    return chain


def create_cohere_rerank_chain(naive_retriever, rag_prompt, chat_model):
    """Create Cohere rerank retrieval chain."""
    print("\n🔀 Creating Cohere Rerank Chain...")
    
    try:
        compressor = CohereRerank(model="rerank-v3.5")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=naive_retriever
        )
        
        chain = (
            {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
        )
        print("✅ Cohere rerank chain created")
        return chain
    except Exception as e:
        print(f"⚠️  Failed to create Cohere rerank chain: {e}")
        return None


def create_multi_query_chain(naive_retriever, rag_prompt, chat_model):
    """Create multi-query retrieval chain."""
    print("\n🔀 Creating Multi-Query Chain...")
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=naive_retriever, llm=chat_model
    )
    
    chain = (
        {"context": itemgetter("question") | multi_query_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    print("✅ Multi-query chain created")
    return chain


def evaluate_retriever_method(method_name, chain, test_set, evaluator_llm, apply_rate_limit=False):
    """Evaluate a retrieval method using RAGAS metrics."""
    print(f"\n{'='*60}")
    print(f"🔬 EVALUATING: {method_name}")
    print(f"{'='*60}")
    start_time = time.time()
    
    # Initialize result dictionary
    result_metrics = {
        'Method': method_name,
        'Status': 'Failed',
        'Evaluation_Time_seconds': 0,
        'context_precision': None,
        'context_recall': None,
        'faithfulness': None,
        'answer_relevancy': None,
        'answer_correctness': None,
        'coherence': None,
        'conciseness': None
    }
    
    try:
        # Generate responses
        responses = []
        contexts = []
        
        print(f"Generating responses for {len(test_set)} questions...")
        for idx, row in test_set.iterrows():
            # Apply rate limit delay if needed (for Cohere)
            if apply_rate_limit and idx > 0:
                print(f"⏳ Rate limit delay (30s) before question {idx+1}...")
                time.sleep(30)
            
            result = chain.invoke({"question": row['user_input']})
            response = result["response"].content
            context = [str(doc.page_content) for doc in result["context"]]
            responses.append(response)
            contexts.append(context)
            print(f"  ✓ Question {idx+1}/{len(test_set)} processed")
        
        # Create evaluation dataset (RAGAS 0.3.7 format)
        eval_df = pd.DataFrame({
            'user_input': test_set['user_input'].tolist(),
            'response': responses,
            'retrieved_contexts': contexts,
            'reference': test_set['reference'].tolist()
        })
        eval_dataset = EvaluationDataset.from_pandas(eval_df)
        
        # Define RAGAS metrics (including answer quality and critique metrics)
        # AspectCritic allows evaluation of specific aspects like coherence and conciseness
        coherence_critic = AspectCritic(name="coherence", definition="Is the response logically consistent and well-structured?")
        conciseness_critic = AspectCritic(name="conciseness", definition="Is the response brief and to the point without unnecessary details?")
        
        ragas_metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,    # Factual correctness
            coherence_critic,      # Logical flow and consistency
            conciseness_critic     # Brevity and efficiency
        ]
        
        # Run RAGAS evaluation (RAGAS 0.3.7 API)
        print(f"\n🔍 Running RAGAS evaluation...")
        ragas_result = evaluate(
            dataset=eval_dataset,
            metrics=ragas_metrics
        )
        
        # Extract scores - ragas_result is a Dataset/dict-like object
        # In RAGAS 0.2.10, we need to convert to pandas and get mean scores
        metric_names = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy', 
                        'answer_correctness', 'coherence', 'conciseness']
        
        if hasattr(ragas_result, 'to_pandas'):
            result_df = ragas_result.to_pandas()
            # Get mean scores for each metric column
            for col in result_df.columns:
                if col in metric_names:
                    result_metrics[col] = result_df[col].mean()
        else:
            # Fallback: try direct attribute access
            for metric_name in metric_names:
                if hasattr(ragas_result, metric_name):
                    result_metrics[metric_name] = getattr(ragas_result, metric_name)
                elif metric_name in ragas_result:
                    result_metrics[metric_name] = ragas_result[metric_name]
        
        elapsed_time = time.time() - start_time
        result_metrics.update({
            'Status': 'Success',
            'Evaluation_Time_seconds': elapsed_time
        })
        
        print(f"\n✅ {method_name} completed in {elapsed_time:.2f} seconds")
        print(f"Results:")
        for key, value in result_metrics.items():
            if key not in ['Method', 'Status', 'Evaluation_Time_seconds'] and value is not None:
                print(f"  - {key}: {value:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error evaluating {method_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        result_metrics['Status'] = 'Failed'
        result_metrics['Evaluation_Time_seconds'] = time.time() - start_time
    
    return result_metrics


def save_best_config(all_results):
    """Save the best retrieval configuration based on average score."""
    # Calculate average scores
    for result in all_results:
        if result['Status'] == 'Success':
            scores = [v for k, v in result.items() 
                     if k not in ['Method', 'Status', 'Evaluation_Time_seconds'] and v is not None]
            result['average_score'] = sum(scores) / len(scores) if scores else 0
        else:
            result['average_score'] = 0
    
    # Find best result
    best_result = max(all_results, key=lambda x: x.get('average_score', 0))
    
    # Save configuration
    config = {
        "best_retriever": best_result["Method"],
        "average_score": best_result["average_score"],
        "metrics": {k: v for k, v in best_result.items() 
                   if k not in ['Method', 'Status', 'Evaluation_Time_seconds', 'average_score']},
        "timestamp": datetime.now().isoformat()
    }
    
    config_path = Path(__file__).parent.parent / "config" / "retrieval_config.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Saved best configuration to {config_path}")
    print(f"   Best retriever: {best_result['Method']}")
    print(f"   Average score: {best_result['average_score']:.3f}")


def main():
    """Main evaluation function - tests all chunking + retrieval combinations."""
    print("🚀 Starting Comprehensive RAG Evaluation with RAGAS\n")
    print("=" * 80)
    print("📊 Testing 2 Chunking Strategies × 3 Retrieval Methods = 6 Total Combinations")
    print("=" * 80)
    
    # Setup
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    # Setup RAGAS components
    generator_llm = ChatOpenAI(model="gpt-4o-mini")
    generator_embeddings = OpenAIEmbeddings()
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Load base documents once (no chunking yet, just load + filter)
    print("\n📚 Loading and filtering documents...")
    loader = DirectoryLoader(str(DATA_DIR), glob="*.pdf", loader_cls=PyMuPDFLoader)
    all_docs = loader.load()
    print(f"✅ Loaded {len(all_docs)} document pages")
    
    # Apply language filter once
    from langdetect import detect, LangDetectException
    filtered_docs = []
    for doc in all_docs:
        try:
            detected_lang = detect(doc.page_content[:500])
            if detected_lang == 'en':
                filtered_docs.append(doc)
        except LangDetectException:
            continue
    print(f"✅ Filtered to {len(filtered_docs)} EN pages (from {len(all_docs)} total)")
    docs = filtered_docs
    
    # Generate test set once (reused for all evaluations)
    test_set = generate_or_load_testset(docs, generator_llm, generator_embeddings)
    
    # Define chunking strategies to test
    chunking_strategies = [
        ('Recursive', 'recursive'),
        ('Semantic', 'semantic')
    ]
    
    # Evaluate all chunking × retrieval combinations
    print("\n" + "=" * 80)
    print("STARTING EVALUATIONS")
    print("=" * 80)
    
    all_results = []
    
    for chunk_name, chunk_strategy in chunking_strategies:
        print(f"\n{'='*80}")
        print(f"🔹 CHUNKING STRATEGY: {chunk_name}")
        print(f"{'='*80}")
        
        # Chunk documents using the appropriate strategy
        if chunk_strategy == 'semantic':
            print(f"🔍 Using SemanticChunker (percentile breakpoint)...")
            text_splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile"
            )
        else:  # recursive
            print(f"📝 Using RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
        
        rag_documents = text_splitter.split_documents(docs)
        print(f"✅ Created {len(rag_documents)} {chunk_strategy} chunks")
        
        # Create vector store for this chunking strategy
        vectorstore = create_vector_store(rag_documents, embeddings)
        
        # Create base retriever
        naive_retriever = create_base_retriever(vectorstore)
        
        # Test all 3 retrieval methods with this chunking strategy
        retrieval_methods = [
            (f"{chunk_name} + Naive Retrieval", lambda: create_naive_retrieval_chain(naive_retriever, rag_prompt, chat_model), False),
            (f"{chunk_name} + Cohere Rerank", lambda: create_cohere_rerank_chain(naive_retriever, rag_prompt, chat_model), True),
            (f"{chunk_name} + Multi-Query", lambda: create_multi_query_chain(naive_retriever, rag_prompt, chat_model), False)
        ]
        
        for method_name, chain_factory, apply_rate_limit in retrieval_methods:
            chain = chain_factory()
            if chain or not apply_rate_limit:  # Cohere might be None if API key missing
                result = evaluate_retriever_method(
                    method_name, chain, test_set, evaluator_llm,
                    apply_rate_limit=apply_rate_limit
                )
                all_results.append(result)
    
    # Display final results
    print("\n" + "=" * 80)
    print("🏆 FINAL RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_path = RESULTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to {results_path}")
    
    # Save best configuration
    save_best_config(all_results)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
