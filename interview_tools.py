import logging
from typing import List
from livekit.agents import function_tool, RunContext
from interview_state import InterviewState

logger = logging.getLogger("interview-coach.tools")


class InterviewTools:
    """
    Function tools exposed to the LLM for interview management.
    These enable the agent to verify facts, score answers, and manage state.
    
    Knowledge base covers 60+ Generative AI terms including:
    - Transformer architecture and attention mechanisms
    - LLM types (GPT, BERT, T5) and architectures
    - Tokenization methods (BPE, WordPiece, SentencePiece)
    - RAG and retrieval systems
    - Fine-tuning techniques (LoRA, QLoRA, PEFT, RLHF)
    - Prompt engineering and sampling strategies
    - Agentic AI (ReAct, tool use, multi-agent)
    - Safety and evaluation
    """
    
    def __init__(self, state: InterviewState):
        self.state = state
        
        # Knowledge base for technical definitions
        self.definitions = {
            # Core Transformer Architecture
            "transformer": (
                "A neural network architecture introduced in 'Attention is All You Need' that uses "
                "self-attention mechanisms to process sequential data in parallel, eliminating the need "
                "for recurrence. Consists of encoder and decoder stacks with multi-head attention, "
                "feed-forward networks, and positional encoding."
            ),
            "attention": (
                "A mechanism that allows models to weigh the importance of different parts of the input "
                "when making predictions. Computes attention scores using queries, keys, and values, "
                "enabling the model to focus on relevant information dynamically."
            ),
            "self-attention": (
                "An attention mechanism where a sequence attends to itself, allowing each position to "
                "attend to all positions in the previous layer. Enables capturing long-range dependencies "
                "and contextual relationships within the input sequence."
            ),
            "multi-head attention": (
                "A mechanism that runs multiple attention operations in parallel, each learning different "
                "representation subspaces. Allows the model to jointly attend to information from different "
                "positions and representation subspaces, improving model expressiveness."
            ),
            "positional encoding": (
                "A technique to inject sequence order information into transformer models, which otherwise "
                "process tokens in parallel without inherent position awareness. Typically uses sinusoidal "
                "functions or learned embeddings added to input embeddings."
            ),
            
            # Language Models
            "llm": (
                "Large Language Model - A neural network trained on massive text corpora (billions to "
                "trillions of tokens) to understand and generate human-like text. Examples include GPT-4, "
                "Claude, Gemini, and LLaMA. Capable of few-shot learning and complex reasoning tasks."
            ),
            "gpt": (
                "Generative Pre-trained Transformer - A decoder-only transformer architecture trained with "
                "causal language modeling (predicting next tokens). Uses unidirectional attention, making it "
                "excellent for text generation. GPT-3 has 175B parameters."
            ),
            "bert": (
                "Bidirectional Encoder Representations from Transformers - An encoder-only model that uses "
                "masked language modeling and next sentence prediction for pre-training. Processes text "
                "bidirectionally, making it excellent for understanding tasks like classification and NER."
            ),
            "t5": (
                "Text-to-Text Transfer Transformer - A model that frames all NLP tasks as text-to-text "
                "problems, using both encoder and decoder. Trained with a denoising objective where "
                "corrupted text spans are predicted."
            ),
            "decoder-only": (
                "An architecture that uses only the decoder portion of transformers with causal (masked) "
                "attention, preventing positions from attending to future tokens. Used by GPT models, "
                "optimized for autoregressive generation."
            ),
            "encoder-only": (
                "An architecture using only the encoder with bidirectional attention, allowing each token "
                "to attend to all others. Used by BERT, optimized for understanding tasks like "
                "classification and embedding generation."
            ),
            
            # Tokenization
            "tokenization": (
                "The process of breaking text into smaller units (tokens) that can be processed by neural "
                "networks. Tokens can be words, subwords, or characters. Essential for converting text "
                "into numerical representations models can process."
            ),
            "bpe": (
                "Byte Pair Encoding - A subword tokenization algorithm that iteratively merges the most "
                "frequent pairs of bytes or characters. Balances vocabulary size with sequence length, "
                "handling rare words through subword units. Used by GPT models."
            ),
            "wordpiece": (
                "A subword tokenization algorithm similar to BPE but uses likelihood-based merging. "
                "Splits unknown words into known subword units. Used by BERT and many Google models. "
                "Provides good balance for multilingual applications."
            ),
            "sentencepiece": (
                "A language-agnostic tokenization library that treats text as a sequence of Unicode "
                "characters and learns subword units directly from raw text without pre-tokenization. "
                "Supports BPE and unigram language models."
            ),
            
            # Embeddings
            "embeddings": (
                "Dense vector representations of tokens, words, or sentences in continuous vector space "
                "where semantic similarity is captured by geometric proximity. Learned during training, "
                "typically 768-4096 dimensions for modern LLMs."
            ),
            "vector database": (
                "A specialized database optimized for storing and querying high-dimensional vectors "
                "(embeddings). Supports similarity search using metrics like cosine similarity or "
                "Euclidean distance. Examples: Pinecone, Weaviate, Milvus, ChromaDB."
            ),
            "cosine similarity": (
                "A metric measuring the cosine of the angle between two vectors, ranging from -1 to 1. "
                "Commonly used to compare embeddings, where values close to 1 indicate high similarity. "
                "Normalized by vector magnitude, making it scale-invariant."
            ),
            
            # RAG (Retrieval Augmented Generation)
            "rag": (
                "Retrieval Augmented Generation - A technique that enhances LLM responses by retrieving "
                "relevant information from external knowledge bases before generation. Combines retrieval "
                "systems with generative models to provide factual, up-to-date, and source-grounded answers."
            ),
            "retrieval": (
                "The process of finding and fetching relevant documents or information from a knowledge base "
                "based on a query. In RAG systems, typically uses semantic search over embeddings to find "
                "documents most relevant to the user's question."
            ),
            "semantic search": (
                "Search that understands the meaning and context of queries rather than just keyword matching. "
                "Uses embeddings to find semantically similar content even with different wording. "
                "More effective than lexical search for conceptual queries."
            ),
            "chunking": (
                "The process of splitting documents into smaller segments for embedding and retrieval. "
                "Balances between preserving context and fitting within model token limits. Common strategies "
                "include fixed-size, sentence-based, or semantic chunking."
            ),
            
            # Fine-tuning
            "fine-tuning": (
                "The process of further training a pre-trained model on domain-specific or task-specific data "
                "to adapt it for particular use cases. Updates model weights through continued training with "
                "smaller learning rates to preserve general knowledge while specializing."
            ),
            "lora": (
                "Low-Rank Adaptation - A parameter-efficient fine-tuning method that freezes the pre-trained "
                "model and injects trainable low-rank decomposition matrices into each layer. Reduces trainable "
                "parameters by 10,000x while maintaining performance, enabling fine-tuning on consumer hardware."
            ),
            "qlora": (
                "Quantized LoRA - Extends LoRA by quantizing the base model to 4-bit precision, further reducing "
                "memory requirements. Enables fine-tuning 65B+ parameter models on a single GPU while maintaining "
                "quality through 16-bit LoRA adapters."
            ),
            "peft": (
                "Parameter-Efficient Fine-Tuning - A family of techniques that fine-tune models by updating "
                "only a small subset of parameters. Includes LoRA, prefix tuning, and adapter layers. "
                "Reduces computational cost and memory while achieving comparable performance."
            ),
            "instruction tuning": (
                "Fine-tuning LLMs on datasets of instruction-response pairs to improve their ability to follow "
                "instructions and perform tasks in a zero-shot manner. Creates more helpful, aligned models. "
                "Used to create models like InstructGPT and Flan-T5."
            ),
            "rlhf": (
                "Reinforcement Learning from Human Feedback - A technique that fine-tunes models using "
                "human preferences as reward signals. Trains a reward model from human comparisons, then "
                "optimizes the LLM using PPO or similar RL algorithms. Used to align models with human values."
            ),
            
            # Prompt Engineering
            "prompt engineering": (
                "The practice of designing and optimizing input prompts to elicit desired behaviors from LLMs. "
                "Includes techniques like few-shot learning, chain-of-thought, and role prompting. "
                "Critical for maximizing model performance without fine-tuning."
            ),
            "few-shot learning": (
                "Providing a few examples (typically 1-10) in the prompt to demonstrate the desired task format "
                "and behavior. Enables models to adapt to new tasks without fine-tuning by learning from "
                "in-context examples. More examples generally improve performance."
            ),
            "zero-shot": (
                "Asking a model to perform a task without providing any examples, relying solely on its "
                "pre-trained knowledge and instruction-following ability. Tests the model's generalization "
                "and understanding of task descriptions."
            ),
            "chain-of-thought": (
                "A prompting technique that encourages models to show their reasoning process step-by-step "
                "before arriving at a final answer. Significantly improves performance on complex reasoning "
                "tasks like math, logic, and multi-step problems."
            ),
            "system prompt": (
                "Initial instructions that set the context, role, and behavioral guidelines for the AI assistant. "
                "Typically includes persona definition, task description, formatting requirements, and constraints. "
                "Persists across the conversation to guide model behavior."
            ),
            
            # Generation & Sampling
            "temperature": (
                "A sampling parameter that controls randomness in text generation. Higher values (e.g., 1.0) "
                "increase creativity and diversity by flattening probability distributions. Lower values "
                "(e.g., 0.1) make output more deterministic and focused. Temperature of 0 is greedy decoding."
            ),
            "top-k sampling": (
                "A sampling strategy that selects the next token from the k most probable tokens. Limits "
                "the sampling pool to avoid very unlikely tokens while maintaining diversity. "
                "Common values: k=40-50. Provides local filtering of improbable options."
            ),
            "top-p sampling": (
                "Also called nucleus sampling - selects from the smallest set of tokens whose cumulative "
                "probability exceeds p (e.g., 0.9). Adapts the selection pool size based on probability "
                "distribution shape. More dynamic than top-k, often preferred for generation quality."
            ),
            "greedy decoding": (
                "A deterministic decoding strategy that always selects the most probable next token. "
                "Equivalent to temperature=0. Produces consistent outputs but can be repetitive and "
                "lacks creativity. Fast but may not produce the most human-like text."
            ),
            "beam search": (
                "A search algorithm that maintains the top k most likely sequences at each step, exploring "
                "multiple paths simultaneously. More thorough than greedy decoding but computationally expensive. "
                "Often used in translation; less common in modern LLM generation."
            ),
            
            # Agentic AI
            "agent": (
                "An AI system that can perceive its environment, make decisions, and take actions to achieve "
                "goals autonomously. LLM agents use language models as reasoning engines, combined with tools "
                "and memory, to complete complex multi-step tasks."
            ),
            "tool use": (
                "Also called function calling - the ability of LLMs to invoke external tools or APIs to extend "
                "their capabilities beyond text generation. Tools can include calculators, databases, web search, "
                "code execution, and custom functions. Critical for agentic systems."
            ),
            "react": (
                "Reasoning and Acting - A prompting framework where agents alternate between reasoning about "
                "what to do next and taking actions (using tools). Uses Thought-Action-Observation loops. "
                "Improves task success by making decision-making explicit and interpretable."
            ),
            "memory": (
                "Systems that allow agents to store and retrieve information across interactions. Includes "
                "short-term (working memory, conversation history) and long-term (vector stores, databases). "
                "Essential for maintaining context and learning from past experiences."
            ),
            "multi-agent": (
                "Systems where multiple AI agents collaborate, each with specialized roles or capabilities. "
                "Agents communicate, delegate tasks, and combine expertise to solve complex problems. "
                "Examples include AutoGen, CrewAI, and LangGraph multi-agent workflows."
            ),
            "planning": (
                "The process by which agents decompose complex goals into actionable sub-tasks and determine "
                "execution strategies. Techniques include task decomposition, chain-of-thought planning, and "
                "tree-of-thought reasoning. Critical for multi-step problem solving."
            ),
            
            # Context & Architecture
            "context window": (
                "The maximum number of tokens (input + output) a model can process in a single forward pass. "
                "Represents the model's \"working memory.\" GPT-4 supports up to 128K tokens. Larger windows "
                "enable processing longer documents but increase computational cost quadratically."
            ),
            "kv cache": (
                "Key-Value cache - An optimization that stores computed attention keys and values from previous "
                "tokens during autoregressive generation. Avoids recomputing attention for all past tokens, "
                "dramatically speeding up inference but increasing memory usage."
            ),
            "flash attention": (
                "An optimized attention algorithm that reduces memory usage and speeds up computation by "
                "reordering operations and minimizing memory reads/writes. Achieves 2-4x speedup without "
                "approximation. Critical for training and serving large models efficiently."
            ),
            
            # Evaluation & Safety
            "perplexity": (
                "A metric measuring how well a language model predicts a sample of text. Lower perplexity "
                "indicates better prediction. Calculated as the exponential of cross-entropy loss. "
                "Common for evaluating language modeling quality but doesn't capture task performance."
            ),
            "hallucination": (
                "When an LLM generates information that sounds plausible but is factually incorrect or entirely "
                "fabricated. Caused by the model's lack of grounding in real knowledge and tendency to pattern "
                "match. Mitigated by RAG, citations, and alignment techniques."
            ),
            "alignment": (
                "The process of ensuring AI systems behave according to human values and intentions. Includes "
                "techniques like RLHF, constitutional AI, and red-teaming. Aims to make models helpful, "
                "harmless, and honest. Critical safety concern for deployed systems."
            ),
            "jailbreaking": (
                "Techniques to bypass safety guardrails and content policies of aligned LLMs. Exploits weaknesses "
                "in instruction-following or reasoning to elicit prohibited outputs. Important consideration "
                "for adversarial robustness and safety testing."
            ),
        }
    
    def get_tools(self) -> List:
        """Return list of tool functions for the agent"""
        return [
            self.get_definition,
            self.score_answer,
            self.submit_final_score,
            self.get_interview_status,
        ]
    
    @function_tool(
        description=(
            "Look up the official definition of a technical term or concept. "
            "Use this to verify if a candidate's answer is correct. "
            "Returns the authoritative definition from the knowledge base."
        )
    )
    async def get_definition(self, context: RunContext, term: str) -> str:
        """
        Retrieve technical definition for verification.
        
        Args:
            term: The technical term to look up (e.g., 'WebRTC', 'SFU')
            
        Returns:
            Official definition or error message
        """
        term_lower = term.lower().strip()
        logger.info(f"Tool call: get_definition('{term}')")
        
        if term_lower in self.definitions:
            definition = self.definitions[term_lower]
            logger.info(f"Definition found for '{term}'")
            return definition
        else:
            logger.warning(f"Definition not found for '{term}'")
            return (
                f"Definition for '{term}' not found in the knowledge base. "
                "Please rephrase or ask the candidate to clarify the term."
            )
    
    @function_tool(
        description=(
            "Record a score for the candidate's answer to the current question. "
            "Use this after evaluating an answer. Score range: 1-10. "
            "Also provide brief feedback on what was good or needs improvement."
        )
    )
    async def score_answer(
        self,
        context: RunContext,
        question: str,
        score: int,
        feedback: str
    ) -> str:
        """
        Score a candidate's answer and provide feedback.
        
        Args:
            question: The question that was asked
            score: Score from 1-10 (10 = excellent)
            feedback: Brief feedback on the answer
            
        Returns:
            Confirmation message
        """
        logger.info(f"Tool call: score_answer(score={score})")
        
        # Validate score
        if not 1 <= score <= 10:
            return "Error: Score must be between 1 and 10."
        
        # Record the score
        self.state.add_score(question, score, feedback)
        
        # Check if we should advance rounds
        if self.state.should_advance_round():
            self.state.next_round()
            return (
                f"Score recorded. Moving to Round {self.state.round}: "
                f"{self.state.get_current_focus()}"
            )
        
        return "Score recorded. Continue with the next question."
    
    @function_tool(
        description=(
            "Submit the final overall assessment when the interview is complete. "
            "Include the overall score (1-10), key strengths, and areas for improvement."
        )
    )
    async def submit_final_score(
        self,
        context: RunContext,
        overall_score: int,
        strengths: str,
        areas_for_improvement: str
    ) -> str:
        """
        Submit the final interview assessment.
        
        Args:
            overall_score: Final score from 1-10
            strengths: Summary of candidate's strengths
            areas_for_improvement: Areas where candidate can improve
            
        Returns:
            Confirmation message
        """
        logger.info(f"Tool call: submit_final_score(score={overall_score})")
        
        if not 1 <= overall_score <= 10:
            return "Error: Overall score must be between 1 and 10."
        
        # Record final assessment
        self.state.finalize_assessment(
            overall_score=overall_score,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement
        )
        
        logger.info("Interview assessment complete")
        return (
            "Final assessment recorded. Thank the candidate and conclude the interview. "
            "A detailed feedback report will be sent to their dashboard."
        )
    
    @function_tool(
        description=(
            "Get the current interview progress and state. "
            "Use this if you need to check what round you're in or how many questions "
            "have been asked."
        )
    )
    async def get_interview_status(self, context: RunContext) -> str:
        """
        Get current interview state information.
        
        Returns:
            JSON string with current state
        """
        import json
        
        status = {
            "current_round": self.state.round,
            "total_rounds": self.state.max_rounds,
            "focus_area": self.state.get_current_focus(),
            "questions_asked": len(self.state.question_history),
            "scores_recorded": len(self.state.scores),
            "is_complete": self.state.is_complete()
        }
        
        logger.info(f"Tool call: get_interview_status() -> {status}")
        return json.dumps(status, indent=2)