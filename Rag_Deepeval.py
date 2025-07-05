#!/usr/bin/env python3
import os
import time
import json
from typing import Iterator, List, Dict, Any
from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase

# Disable detailed logs
set_debug(False)
set_verbose(False)

PDF_FILE_PATH = "attention.pdf"

class ChatPDF:
    def __init__(self, llm_model: str = "mistral"):
        # Enable streaming in the model
        self.model = ChatOllama(model=llm_model, temperature=0.2, streaming=True)
        # Non-streaming model for evaluation
        self.eval_model = ChatOllama(model=llm_model, temperature=0.2, streaming=False)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        # Update the prompt to instruct the model to provide direct answers
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a smart and professional question-answering assistant. "
                    "You respond clearly and accurately using only the information provided. "
                    "Give brief but informative answers, including helpful details when necessary. "
                    "Do not reference or mention 'the context' or any documents. "
                    "If the answer is not available in the provided content, politely say so and offer further help.\n\n"
                    "Context:\n{context}"
                ),
                (
                    "human",
                    "Question: {question}"
                )
,
            ]
        )
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.eval_chain = None
        self.retrieved_contexts = []  # Store contexts for evaluation

    def ingest(self, pdf_file_path: str):
        """Ingest PDF document and create vector store"""
        print("Ingesting PDF document...")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )
        print("Ingestion completed.\n")

    def ask(self, query: str) -> Iterator[str]:
        """Ask question with streaming response"""
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return self.chain.stream(query)

    def ask_eval(self, query: str) -> Dict[str, Any]:
        """Ask question for evaluation purposes (non-streaming)"""
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        # Get retrieved contexts
        retrieved_docs = self.retriever.get_relevant_documents(query)
        contexts = [doc.page_content for doc in retrieved_docs]
        
        # Create evaluation chain
        self.eval_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.eval_model
            | StrOutputParser()
        )

        # Get answer
        answer = self.eval_chain.invoke(query)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_docs": retrieved_docs
        }

    def clear(self):
        """Clear the vector store and chains"""
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.eval_chain = None

class RAGEvaluator:
    def __init__(self, chat_pdf: ChatPDF):
        self.chat_pdf = chat_pdf
        self.test_cases = []
        
    def create_test_dataset(self) -> List[Dict[str, str]]:
        """Create a comprehensive test dataset for evaluation"""
        test_questions =[
            {
                "question": "What is the role of attention in transformer models?",
                "expected_answer": "Attention allows transformer models to weigh the importance of different input tokens dynamically, enabling them to focus on relevant parts of the sequence.",
                "context": "transformers, attention mechanism, input relevance"
            },
            {
                "question": "What are the main components of the transformer architecture?",
                "expected_answer": "The transformer architecture includes multi-head self-attention, position-wise feedforward networks, layer normalization, and residual connections.",
                "context": "transformer components, architecture design, self-attention layers"
            },
            {
                "question": "How does self-attention differ from traditional attention mechanisms?",
                "expected_answer": "Self-attention computes attention scores within a single sequence, allowing each token to attend to all others, unlike traditional attention that typically aligns inputs and outputs.",
                "context": "self-attention vs traditional attention, attention computation"
            },
            {
                "question": "Why is positional encoding used in transformers?",
                "expected_answer": "Since transformers lack recurrence, positional encoding provides information about token order, enabling the model to understand sequence structure.",
                "context": "positional encoding, transformer sequence order"
            },
            {
                "question": "What is multi-head attention and why is it important?",
                "expected_answer": "Multi-head attention allows the model to focus on different parts of the input simultaneously, capturing diverse patterns and relationships.",
                "context": "multi-head attention, parallel attention heads"
            },
            {
                "question": "How are transformers trained on large datasets?",
                "expected_answer": "Transformers are trained using large-scale parallel processing with techniques like masked language modeling or autoregressive prediction and optimized using gradient descent.",
                "context": "transformer training, large-scale learning, optimization"
            },
            {
                "question": "What is the difference between encoder and decoder in transformers?",
                "expected_answer": "The encoder processes input sequences into contextual representations, while the decoder generates output sequences using encoder outputs and self-attention.",
                "context": "transformer encoder, transformer decoder, architecture roles"
            },
            {
                "question": "How do transformers achieve parallelization during training?",
                "expected_answer": "Transformers enable parallelization by processing all tokens simultaneously using matrix operations, unlike RNNs that process sequentially.",
                "context": "transformer parallelism, training efficiency"
            },
            {
                "question": "What are the common use cases of transformer models?",
                "expected_answer": "Transformers are widely used in machine translation, text summarization, question answering, and large language models like GPT and BERT.",
                "context": "transformer applications, NLP tasks, LLMs"
            },
            {
                "question": "How does layer normalization help in transformer training?",
                "expected_answer": "Layer normalization stabilizes and accelerates training by normalizing inputs across features, reducing internal covariate shift.",
                "context": "layer normalization, training stability, transformer layers"
            }
        ]

        return test_questions

    def generate_test_cases(self) -> List[LLMTestCase]:
        """Generate LLMTestCase objects for evaluation"""
        test_data = self.create_test_dataset()
        test_cases = []
        
        print("Generating answers for test cases...")
        for i, test_item in enumerate(test_data):
            print(f"Processing test case {i+1}/{len(test_data)}: {test_item['question'][:50]}...")
            
            # Get answer and context from RAG system
            result = self.chat_pdf.ask_eval(test_item['question'])
            
            # Create test case
            test_case = LLMTestCase(
                input=test_item['question'],
                actual_output=result['answer'],
                expected_output=test_item['expected_answer'],
                retrieval_context=result['contexts']
            )
            test_cases.append(test_case)
            
        self.test_cases = test_cases
        return test_cases

    def evaluate_rag_system(self):
        """Comprehensive evaluation of the RAG system"""
        print("\n" + "="*80)
        print("STARTING RAG SYSTEM EVALUATION")
        print("="*80)
        
        # Generate test cases
        test_cases = self.generate_test_cases()
        
        # Define evaluation metrics (using local models where possible)
        try:
            # Try to use metrics that don't require external API
            metrics = [
                AnswerRelevancyMetric(threshold=0.7, model=self.chat_pdf.eval_model),
                FaithfulnessMetric(threshold=0.7, model=self.chat_pdf.eval_model),
                ContextualRelevancyMetric(threshold=0.7, model=self.chat_pdf.eval_model),
            ]
            print("Using local Ollama model for evaluation metrics")
        except Exception as e:
            print(f"Error configuring metrics with local model: {e}")
            # Fallback to simpler metrics or custom evaluation
            metrics = []
            print("Falling back to custom evaluation metrics")
        
        print(f"\nEvaluating {len(test_cases)} test cases with {len(metrics)} metrics...")
        
        # Run evaluation
        if metrics:
            try:
                results = evaluate(test_cases, metrics)
                # Display results
                self.display_evaluation_results(results, metrics)
            except Exception as e:
                print(f"Error running DeepEval metrics: {e}")
                print("Running custom evaluation instead...")
                results = self.run_custom_evaluation(test_cases)
        else:
            results = self.run_custom_evaluation(test_cases)
        
        # Save results
        self.save_evaluation_results(results, test_cases, metrics)
        
        return results
    
    def run_custom_evaluation(self, test_cases):
        """Run custom evaluation when DeepEval metrics fail"""
        print("\n" + "="*50)
        print("CUSTOM RAG EVALUATION")
        print("="*50)
        
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\nEvaluating Test Case {i+1}:")
            print(f"Question: {test_case.input}")
            
            # Simple relevance check (keyword matching)
            relevance_score = self.calculate_relevance_score(test_case.input, test_case.actual_output)
            
            # Context utilization check
            context_score = self.calculate_context_utilization(test_case.actual_output, test_case.retrieval_context)
            
            # Response completeness
            completeness_score = self.calculate_completeness_score(test_case.actual_output)
            
            # Overall score
            overall_score = (relevance_score + context_score + completeness_score) / 3
            
            result = {
                "test_case_id": i + 1,
                "question": test_case.input,
                "relevance_score": relevance_score,
                "context_utilization_score": context_score,
                "completeness_score": completeness_score,
                "overall_score": overall_score,
                "passed": overall_score >= 0.7
            }
            results.append(result)
            
            print(f"  Relevance Score: {relevance_score:.2f}")
            print(f"  Context Utilization: {context_score:.2f}")
            print(f"  Completeness: {completeness_score:.2f}")
            print(f"  Overall Score: {overall_score:.2f}")
            print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
        
        # Summary
        passed_tests = sum(1 for r in results if r['passed'])
        total_tests = len(results)
        pass_rate = (passed_tests / total_tests) * 100
        
        print(f"\n" + "="*50)
        print("CUSTOM EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed Tests: {passed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        return results
    
    def calculate_relevance_score(self, question, answer):
        """Calculate relevance score based on keyword overlap"""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        question_words = question_words - stop_words
        answer_words = answer_words - stop_words
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words.intersection(answer_words))
        return min(overlap / len(question_words), 1.0)
    
    def calculate_context_utilization(self, answer, contexts):
        """Calculate how well the answer utilizes the retrieved context"""
        if not contexts:
            return 0.0
        
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for context in contexts:
            context_words.update(context.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        
        answer_words = answer_words - stop_words
        context_words = context_words - stop_words
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        return min(overlap / len(answer_words), 1.0)
    
    def calculate_completeness_score(self, answer):
        """Calculate completeness based on answer length and structure"""
        if not answer:
            return 0.0
        
        # Basic completeness heuristics
        word_count = len(answer.split())
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        
        # Scoring based on length and structure
        if word_count < 5:
            return 0.2
        elif word_count < 15:
            return 0.5
        elif word_count < 30:
            return 0.7
        else:
            return min(0.9, 0.7 + (sentence_count * 0.05))
    

    def display_evaluation_results(self, results, metrics):
        """Display evaluation results in a formatted way"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        # Check if results are from custom evaluation
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            # Custom evaluation results
            total_tests = len(results)
            passed_tests = sum(1 for r in results if r.get('passed', False))
            pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            print(f"Total Test Cases: {total_tests}")
            print(f"Passed Test Cases: {passed_tests}")
            print(f"Overall Pass Rate: {pass_rate:.1f}%")
            
            # Average scores
            avg_relevance = sum(r.get('relevance_score', 0) for r in results) / len(results)
            avg_context = sum(r.get('context_utilization_score', 0) for r in results) / len(results)
            avg_completeness = sum(r.get('completeness_score', 0) for r in results) / len(results)
            avg_overall = sum(r.get('overall_score', 0) for r in results) / len(results)
            
            print(f"\nAverage Scores:")
            print(f"  Relevance: {avg_relevance:.2f}")
            print(f"  Context Utilization: {avg_context:.2f}")
            print(f"  Completeness: {avg_completeness:.2f}")
            print(f"  Overall: {avg_overall:.2f}")
            
        else:
            # DeepEval results
            total_tests = len(self.test_cases)
            print(f"Total Test Cases: {total_tests}")
            
            # Metric-wise results
            for metric in metrics:
                metric_name = metric.__class__.__name__
                print(f"\n{metric_name}:")
                print(f"  Threshold: {metric.threshold}")
                
                # Calculate pass rate for this metric
                passed = sum(1 for test_case in self.test_cases if hasattr(test_case, 'success') and test_case.success)
                pass_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
                print(f"  Pass Rate: {pass_rate:.1f}%")
        
        # Individual test case results
        print("\n" + "-"*80)
        print("INDIVIDUAL TEST CASE RESULTS")
        print("-"*80)
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\nTest Case {i+1}:")
            print(f"Question: {test_case.input}")
            print(f"Expected: {test_case.expected_output[:100]}...")
            print(f"Actual: {test_case.actual_output[:100]}...")
            print(f"Context Retrieved: {len(test_case.retrieval_context)} chunks")

    def save_evaluation_results(self, results, test_cases, metrics):
        """Save evaluation results to JSON file"""
        evaluation_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(test_cases),
            "evaluation_type": "custom" if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict) else "deepeval",
            "metrics_used": [metric.__class__.__name__ for metric in metrics] if metrics else ["Custom Relevance", "Custom Context Utilization", "Custom Completeness"],
            "test_cases": []
        }
        
        # Handle custom evaluation results
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            for i, (test_case, result) in enumerate(zip(test_cases, results)):
                case_data = {
                    "id": i + 1,
                    "question": test_case.input,
                    "expected_answer": test_case.expected_output,
                    "actual_answer": test_case.actual_output,
                    "context_chunks": len(test_case.retrieval_context),
                    "context_preview": test_case.retrieval_context[0][:200] + "..." if test_case.retrieval_context else "No context",
                    "scores": {
                        "relevance": result.get('relevance_score', 0),
                        "context_utilization": result.get('context_utilization_score', 0),
                        "completeness": result.get('completeness_score', 0),
                        "overall": result.get('overall_score', 0)
                    },
                    "passed": result.get('passed', False)
                }
                evaluation_data["test_cases"].append(case_data)
        else:
            # Handle DeepEval results
            for i, test_case in enumerate(test_cases):
                case_data = {
                    "id": i + 1,
                    "question": test_case.input,
                    "expected_answer": test_case.expected_output,
                    "actual_answer": test_case.actual_output,
                    "context_chunks": len(test_case.retrieval_context),
                    "context_preview": test_case.retrieval_context[0][:200] + "..." if test_case.retrieval_context else "No context"
                }
                evaluation_data["test_cases"].append(case_data)
        
        # Save to file
        with open("rag_evaluation_results.json", "w") as f:
            json.dump(evaluation_data, f, indent=2)
        
        print(f"\nEvaluation results saved to 'rag_evaluation_results.json'")

    def analyze_retrieval_quality(self):
        """Analyze the quality of document retrieval"""
        print("\n" + "="*80)
        print("RETRIEVAL QUALITY ANALYSIS")
        print("="*80)
        
        if not self.test_cases:
            print("No test cases available for analysis")
            return
        
        # Analyze context retrieval
        context_lengths = []
        for test_case in self.test_cases:
            context_lengths.append(len(test_case.retrieval_context))
        
        avg_context_length = sum(context_lengths) / len(context_lengths)
        print(f"Average contexts retrieved per query: {avg_context_length:.1f}")
        print(f"Min contexts retrieved: {min(context_lengths)}")
        print(f"Max contexts retrieved: {max(context_lengths)}")
        
        # Analyze context relevance (simplified)
        relevant_contexts = 0
        total_contexts = sum(context_lengths)
        
        print(f"Total contexts analyzed: {total_contexts}")
        print(f"Average context length: {avg_context_length:.1f} chunks per query")

def main():
    """Main function to run the complete RAG evaluation"""
    print("="*80)
    print("RAG SYSTEM EVALUATION FRAMEWORK")
    print("="*80)
    
    # Check if PDF file exists
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: File '{PDF_FILE_PATH}' not found.")
        print("Please ensure the PDF file is in the current directory.")
        return
    
    # Initialize ChatPDF system
    print("Initializing RAG system...")
    chat_pdf = ChatPDF()
    
    # Ingest PDF
    print("Starting document ingestion...")
    t0 = time.time()
    chat_pdf.ingest(PDF_FILE_PATH)
    ingestion_time = time.time() - t0
    print(f"Document ingested successfully in {ingestion_time:.2f} seconds")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(chat_pdf)
    
    # Run comprehensive evaluation
    try:
        evaluation_results = evaluator.evaluate_rag_system()
        
        # Additional analysis
        evaluator.analyze_retrieval_quality()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("Results have been saved to 'rag_evaluation_results.json'")
        print("Review the results to understand your RAG system's performance.")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("This might be due to missing dependencies or API keys.")
        print("Please ensure DeepEval is properly installed and configured.")
    
    # Interactive mode (optional)
    print("\n" + "-"*80)
    print("INTERACTIVE MODE")
    print("-"*80)
    print("You can now test the RAG system interactively.")
    print("Type 'exit' to quit, 'eval' to run evaluation again.")
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            elif query.lower() == "eval":
                evaluator.evaluate_rag_system()
                continue
            
            if not query:
                continue
            
            print("\nAnswer:")
            try:
                for chunk in chat_pdf.ask(query):
                    print(chunk, end="", flush=True)
                    time.sleep(0.02)
                print("\n" + "-"*50)
            except Exception as e:
                print(f"Error: {str(e)}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()