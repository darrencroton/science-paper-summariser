"""
LLM Provider Interface and Implementations for Science Paper Summariser

This module provides a consistent interface for different LLM providers
including Claude, OpenAI, Perplexity, Gemini, and Ollama.

Each provider implements the LLMProvider base class, allowing the main application
to interact with different LLM services through a unified interface.

Classes:
    - LLMProvider: Base abstract class defining the provider interface
    - ClaudeProvider: Implementation for Anthropic's Claude API
    - OpenAIProvider: Implementation for OpenAI's API
    - PerplexityProvider: Implementation for Perplexity AI's API
    - GeminiProvider: Implementation for Google's Gemini API
    - OllamaProvider: Implementation for local Ollama models
"""

import os
import base64
import requests
import time
import logging
import json
from typing import Dict, Any, Tuple, Optional, List, Union


class LLMProvider:
    """
    Base interface for LLM providers.
    
    This abstract class defines the common interface that all LLM providers must implement.
    Each provider handles authentication, API communication, and model-specific parameters
    for its respective service.
    
    Attributes:
        config (dict): Configuration dictionary with provider-specific settings
    """
    
    def __init__(self, config=None):
        """
        Initialize the provider with optional configuration.
        
        Args:
            config (dict, optional): Configuration settings for the provider.
                                    Can include model name, temperature, etc.
        """
        self.config = config or {}
        self.setup()
    
    def setup(self):
        """
        Initialize provider-specific components.
        
        This method should be overridden by subclasses to perform
        necessary setup like API client initialization.
        """
        pass
        
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """
        Process document with the LLM - to be implemented by subclasses.
        
        Args:
            content: The document content (either text or binary PDF data)
            is_pdf (bool): Whether the content is a PDF document
            system_prompt (str): Instructions for the LLM system role
            user_prompt (str): User instructions and content for the LLM
            max_tokens (int): Maximum response tokens to generate
            
        Returns:
            str: The generated summary from the LLM
            
        Raises:
            NotImplementedError: This abstract method must be implemented by subclasses
        """
        raise NotImplementedError
    
    def get_max_context_size(self):
        """
        Return maximum context size for this provider.
        
        Returns:
            int: Maximum context window size in tokens
            
        Raises:
            NotImplementedError: This abstract method must be implemented by subclasses
        """
        raise NotImplementedError
        
    def get_default_model(self):
        """
        Return the default model for this provider.
        
        Returns:
            str: Name of the default model for this provider
            
        Raises:
            NotImplementedError: This abstract method must be implemented by subclasses
        """
        raise NotImplementedError

    def supports_direct_pdf(self):
        """
        Return whether this provider can process PDFs directly without text extraction.
        
        Returns:
            bool: True if the provider can accept raw PDF data, False otherwise
        """
        return False  # Default to False for safety


class ClaudeProvider(LLMProvider):
    """
    Claude/Anthropic API provider implementation.
    
    Handles communication with Anthropic's Claude API, which supports direct
    PDF processing and has large context windows.
    """

    def supports_direct_pdf(self):
        """Claude models support direct PDF processing"""
        return True
    
    def setup(self):
        """
        Initialize Claude client using the Anthropic API key.
        
        Raises:
            ValueError: If ANTHROPIC_API_KEY environment variable is not set
            ImportError: If the anthropic package is not installed
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("The 'anthropic' package is required. Install with: pip install anthropic")
            
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = self.config.get("model", self.get_default_model())
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """
        Process document with Claude API.
        
        Uses Anthropic's Messages API which supports direct PDF document uploads.
        
        Args:
            content: Document content (text or PDF bytes)
            is_pdf (bool): Whether the content is a PDF document
            system_prompt (str): Instructions for the Claude system
            user_prompt (str): User instructions and context
            max_tokens (int): Maximum response length in tokens
            
        Returns:
            str: The generated summary from Claude
        """
        # Create the base message structure
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        }]
        
        # If content is a PDF, add it as a document
        if is_pdf and content:
            messages[0]["content"].append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(content).decode()
                }
            })
            
        # Make the API request
        message = self.client.messages.create(
            model=self.model,
            temperature=self.config.get("temperature", 0.2),
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        )
        
        return message.content[0].text
    
    def get_max_context_size(self):
        """
        Return maximum context size for Claude model.
        
        Returns:
            int: Maximum context window size in tokens for the current model
        """
        model_contexts = {
            "claude-3-7-sonnet-20250219": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000
        }
        return model_contexts.get(self.model, 200000)
    
    def get_default_model(self):
        """
        Return the default Claude model.
        
        Returns:
            str: Name of the default Claude model to use
        """
        return "claude-3-7-sonnet-20250219"


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider implementation.
    
    Handles communication with OpenAI's API, supporting both standard text
    completion and direct PDF processing via the Responses API.
    """

    def supports_direct_pdf(self):
        """OpenAI's newer models support direct PDF processing via the Responses API"""
        return True

    def setup(self):
        """
        Initialize OpenAI client using the OpenAI API key.
        
        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
            ImportError: If the openai package is not installed
        """
        try:
            import openai
        except ImportError:
            raise ImportError("The 'openai' package is required. Install with: pip install openai")
            
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = self.config.get("model", self.get_default_model())
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """
        Process document with OpenAI API.
        
        Uses OpenAI's Responses API for PDF documents and Chat Completions API for text.
        
        Args:
            content: Document content (text or PDF bytes)
            is_pdf (bool): Whether the content is a PDF document
            system_prompt (str): Instructions for the system role
            user_prompt (str): User instructions and context
            max_tokens (int): Maximum response length in tokens
            
        Returns:
            str: The generated summary from OpenAI
        """
        # For PDFs, use the Responses API which supports direct PDF input
        if is_pdf and isinstance(content, bytes):
            # Creating input structure for Responses API
            input_content = [
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {
                            "type": "input_file",
                            "file_data": f"data:application/pdf;base64,{base64.b64encode(content).decode('utf-8')}",
                            "filename": "document.pdf"
                        }
                    ]
                }
            ]
            
            # Use the Responses API for PDF handling
            # Responses API doesn't use max_tokens or max_completion_tokens
            response = self.client.responses.create(
                model=self.model,
                input=input_content,
                temperature=self.config.get("temperature", 0.2)
            )
            
            # Extract the text from the response
            return response.output[0].content[0].text
            
        else:
            # For regular text content, use the Chat Completions API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Use the Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.get("temperature", 0.2),
                max_tokens=max_tokens
            )
        
            return response.choices[0].message.content
    
    def get_max_context_size(self):
        """
        Return maximum context size for OpenAI model.
        
        Returns:
            int: Maximum context window size in tokens for the current model
        """
        model_contexts = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000, 
            "gpt-4.1": 1047576, 
            "gpt-4.1-mini": 1047576, 
            "gpt-4.1-nano": 1047576
        }
        return model_contexts.get(self.model, 16385)
    
    def get_default_model(self):
        """
        Return the default OpenAI model for scientific PDF summarization.
        
        Returns:
            str: Name of the default OpenAI model to use
        """
        return "gpt-4.1"


class PerplexityProvider(LLMProvider):
    """
    Perplexity API provider implementation.
    
    Handles communication with Perplexity AI's API. Note that Perplexity
    currently does not support direct PDF processing.
    """

    def supports_direct_pdf(self):
        """Perplexity API does not currently support direct PDF input"""
        return False

    def setup(self):
        """
        Initialize Perplexity settings using the Perplexity API key.
        
        Raises:
            ValueError: If PERPLEXITY_API_KEY environment variable is not set
        """
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")
            
        self.model = self.config.get("model", self.get_default_model())
        self.api_url = "https://api.perplexity.ai/chat/completions"
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """
        Process document with Perplexity API.
        
        Uses Perplexity's API which follows OpenAI-compatible format.
        Note: Perplexity does not support direct PDF processing.
        
        Args:
            content: Document content (must be text, not PDF bytes)
            is_pdf (bool): Whether the content is a PDF document
            system_prompt (str): Instructions for the system role
            user_prompt (str): User instructions and context
            max_tokens (int): Maximum response length in tokens
            
        Returns:
            str: The generated summary from Perplexity
            
        Raises:
            ValueError: If PDF content is not properly extracted
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # For PDFs, text must be extracted before calling this method
        if is_pdf and isinstance(content, bytes):
            # This would be an implementation error, as read_input_file should extract text for Perplexity
            raise ValueError("PDF content needs text extraction before using with Perplexity API")
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logging.error(f"Perplexity API error: {str(e)}")
            # Add more detailed error information if available
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logging.error(f"Perplexity response: {e.response.text}")
            raise
        
        # Parse response
        try:
            response_data = response.json()
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                logging.error(f"Unexpected Perplexity response format: {response_data}")
                raise ValueError("No response content received from Perplexity")
        except ValueError as e:
            logging.error(f"Failed to parse Perplexity response: {response.text}")
            raise
            
        return response_data["choices"][0]["message"]["content"]
    
    def get_max_context_size(self):
        """
        Return maximum context size for Perplexity model.
        
        Returns:
            int: Maximum context window size in tokens for the current model
        """
        model_contexts = {
            "sonar": 128000,
            "sonar-pro": 128000,
            "sonar-deep-research": 128000,
            "sonar-reasoning": 128000,
            "sonar-reasoning-pro": 128000,
            "r1-1776": 128000
        }
        return model_contexts.get(self.model, 128000)
    
    def get_default_model(self):
        """
        Return the default Perplexity model.
        
        Returns:
            str: Name of the default Perplexity model to use
        """
        return "r1-1776"


class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider implementation.
    
    Handles communication with Google's Generative AI API for Gemini models,
    supporting direct PDF processing with newer models.
    """
    
    def supports_direct_pdf(self):
        """Gemini 2.5 Pro and newer support direct PDF input"""
        return True
    
    def setup(self):
        """
        Initialize Google AI API client using the Google API key.
        
        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set
            ImportError: If the google-generativeai package is not installed
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("The 'google-generativeai' package is required. Install with: pip install google-generativeai")
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure the Gemini API with the API key
        genai.configure(api_key=self.api_key)
        
        # Store the genai module reference for later use
        self.genai = genai
        
        # Set the model based on config or default
        self.model = self.config.get("model", self.get_default_model())
        
        # Initialize the model
        try:
            self.client = genai.GenerativeModel(model_name=self.model)
            logging.info(f"Initialized Gemini model: {self.model}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini model: {str(e)}")
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """
        Process document with Google Gemini API.
        
        Uses Gemini's API to process either text or direct PDF content.
        
        Args:
            content: Document content (text or PDF bytes)
            is_pdf (bool): Whether the content is a PDF document
            system_prompt (str): Instructions for the system role
            user_prompt (str): User instructions and context
            max_tokens (int): Maximum response length in tokens
            
        Returns:
            str: The generated summary from Gemini
            
        Raises:
            ValueError: If no content is returned from the API
        """
        try:
            # For debugging
            logging.info(f"Processing with Gemini, PDF: {is_pdf}, Content type: {type(content).__name__}")
            
            # Create the generation config
            generation_config = {
                "temperature": self.config.get("temperature", 0.2),
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Create combined prompt with system instructions
            # Gemini doesn't have separate system and user messages like other APIs
            combined_prompt = f"System instructions: {system_prompt}\n\n{user_prompt}"
            
            # Direct approach without chat history
            if is_pdf and isinstance(content, bytes) and self.supports_direct_pdf():
                logging.info(f"Using direct PDF upload to Gemini ({len(content)} bytes)")
                
                # Create content parts with text and PDF
                content_parts = [
                    {"text": combined_prompt},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": base64.b64encode(content).decode('utf-8')
                        }
                    }
                ]
                
                # Generate content with PDF
                response = self.client.generate_content(
                    contents=content_parts,
                    generation_config=generation_config
                )
            else:
                # Generate content with just text
                logging.info("Using text-only prompt with Gemini")
                response = self.client.generate_content(
                    contents=combined_prompt,
                    generation_config=generation_config
                )
            
            # Extract text from response
            result = response.text
            
            if not result:
                raise ValueError("No content returned from Gemini API")
                
            logging.info(f"Successfully received response from Gemini ({len(result)} chars)")
            return result
            
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            raise
    
    def get_max_context_size(self):
        """
        Return maximum context size for Gemini model.
        
        Returns:
            int: Maximum context window size in tokens for the current model
        """
        model_contexts = {
            "gemini-2.5-pro-exp-03-25": 1048576, 
            "gemini-2.0-flash": 1048576,
            "gemini-1.5-pro": 1048576
        }
        return model_contexts.get(self.model, 32768)  # Default to 32K if unknown
    
    def get_default_model(self):
        """
        Return the default Gemini model.
        
        Returns:
            str: Name of the default Gemini model to use
        """
        return "gemini-2.5-pro-exp-03-25"


class OllamaProvider(LLMProvider):
    """
    Ollama API provider implementation.
    
    Handles communication with a local Ollama server for running
    open-source models locally.
    """
    
    def supports_direct_pdf(self):
        """Ollama does not support direct PDF processing"""
        return False

    def setup(self):
        """Initialize Ollama settings for the local Ollama server"""
        self.model = self.config.get("model", self.get_default_model())
        self.base_url = self.config.get("base_url", "http://localhost:11434")
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """
        Process document with Ollama API.
        
        Communicates with a local Ollama server to generate summaries.
        Note: Ollama does not support direct PDF processing.
        
        Args:
            content: Document content (must be text, not PDF bytes)
            is_pdf (bool): Whether the content is a PDF document
            system_prompt (str): Instructions for the system role
            user_prompt (str): User instructions and context
            max_tokens (int): Maximum response length in tokens
            
        Returns:
            str: The generated summary from Ollama
            
        Raises:
            ValueError: If an error is returned from the Ollama API
        """
        # Get the maximum context size for the model
        max_context = self.get_max_context_size()
        
        # Prepare the request data
        data = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.get("temperature", 0.2),
                "num_ctx": min(max_context, 24576),  # Cap at 24K if model supports more
                "num_predict": max_tokens,
                "stop": ["</input>", "</task>"]  # Stop tokens to prevent unnecessary output
            }
        }

        # Make the API call to the local Ollama server
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data
        )
        response.raise_for_status()
                        
        response_data = response.json()
        summary_content = response_data.get('response', '')

        # Check for errors in the response
        if 'error' in response_data:
            raise ValueError(f"Ollama error: {response_data['error']}")
            
        if not summary_content:
            raise ValueError("No content received from Ollama")
            
        return summary_content
    
    def get_max_context_size(self):
        """
        Return maximum context size for Ollama model.
        
        Returns:
            int: Maximum context window size in tokens for the current model
        """
        # Note: This would ideally query the Ollama API for the model's context size
        model_contexts = {
            "qwen2.5:14b-instruct-q8_0": 32768, 
            "llama3.1:8b-instruct-q8_0": 131072
        }
        return model_contexts.get(self.model.lower(), 8192)
    
    def get_default_model(self):
        """
        Return the default Ollama model.
        
        Returns:
            str: Name of the default Ollama model to use
        """
        return "qwen2.5:14b-instruct-q8_0"


def create_llm_provider(provider_name, config=None):
    """
    Create an LLM provider instance based on the provider name.
    
    Factory function that instantiates the appropriate LLM provider class.
    
    Args:
        provider_name (str): Name of the provider to create ('claude', 'openai', etc.)
        config (dict, optional): Configuration dictionary for the provider
        
    Returns:
        LLMProvider: An instance of the requested provider
        
    Raises:
        ValueError: If the requested provider is not supported
    """
    providers = {
        "claude": ClaudeProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "perplexity": PerplexityProvider,
        "gemini": GeminiProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {', '.join(providers.keys())}")
    
    return providers[provider_name](config)
