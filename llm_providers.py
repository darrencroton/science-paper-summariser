"""
LLM Provider Interface and Implementations for Science Paper Summariser

This module provides a consistent interface for different LLM providers
including Claude, OpenAI, Perplexity, Gemini, and Ollama.
"""

import os
import base64
import requests
import time
import logging
import json
from typing import Dict, Any, Tuple, Optional, List, Union

# Do not set up logging here as it will interfere with the application's logging configuration


class LLMProvider:
    """Base interface for LLM providers"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.setup()
    
    def setup(self):
        """Initialize provider-specific components"""
        pass
        
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """Process document with the LLM - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_max_context_size(self):
        """Return maximum context size for this provider"""
        raise NotImplementedError
        
    def get_default_model(self):
        """Return the default model for this provider"""
        raise NotImplementedError

    def supports_direct_pdf(self):
        """Return whether this provider can process PDFs directly without text extraction"""
        return False  # Default to False for safety


class ClaudeProvider(LLMProvider):
    """Claude/Anthropic API provider"""

    def supports_direct_pdf(self):
        return True
    
    def setup(self):
        """Initialize Claude client"""
        import anthropic
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = self.config.get("model", self.get_default_model())
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """Process document with Claude API"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        }]
        
        if is_pdf and content:
            messages[0]["content"].append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(content).decode()
                }
            })
            
        message = self.client.messages.create(
            model=self.model,
            temperature=self.config.get("temperature", 0.2),
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        )
        
        return message.content[0].text
    
    def get_max_context_size(self):
        """Return maximum context size for Claude model"""
        model_contexts = {
            "claude-3-7-sonnet-20250219": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000
        }
        return model_contexts.get(self.model, 200000)
    
    def get_default_model(self):
        """Return the default Claude model"""
        return "claude-3-7-sonnet-20250219"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def supports_direct_pdf(self):
        return True

    def setup(self):
        """Initialize OpenAI client"""
        import openai
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = self.config.get("model", self.get_default_model())
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """Process document with OpenAI API"""
        
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
        """Return maximum context size for OpenAI model"""
        model_contexts = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000, 
            "gpt-4.1": 1047576, 
            "gpt-4.1-mini": 1047576, 
            "gpt-4.1-nano": 1047576
        }
        return model_contexts.get(self.model, 16385)
    
    def get_default_model(self):
        """Return the default OpenAI model for scientific PDF summarization"""
        return "gpt-4.1"


class PerplexityProvider(LLMProvider):
    """Perplexity API provider"""

    def supports_direct_pdf(self):
        return False

    def setup(self):
        """Initialize Perplexity settings"""
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")
            
        self.model = self.config.get("model", self.get_default_model())
        self.api_url = "https://api.perplexity.ai/chat/completions"
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """Process document with Perplexity API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # For PDFs, we use the extracted text which should already be handled in read_input_file
        # If content is binary (PDF data), it means text wasn't extracted yet
        if is_pdf and isinstance(content, bytes):
            # This shouldn't happen, as read_input_file should have extracted text for Perplexity
            # But if it does, let's raise a clear error
            raise ValueError("PDF content needs text extraction before using with Perplexity API")
        
        data = {
            "model": self.model,  # Perplexity expects specific model identifiers
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
        """Return maximum context size for Perplexity model"""
        # Perplexity models and their context sizes 
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
        """Return the default Perplexity model"""
        return "r1-1776"


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""
    
    def supports_direct_pdf(self):
        # Gemini 2.5 Pro supports direct PDF input
        return True
    
    def setup(self):
        """Initialize Google AI API client"""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("The 'google-generativeai' package is required. Install with: pip install google-generativeai")
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure the Gemini API with your API key
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
        """Process document with Google Gemini API"""
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
        """Return maximum context size for Gemini model"""
        # Gemini models and their context sizes
        model_contexts = {
            "gemini-2.5-pro-exp-03-25": 1048576, 
            "gemini-2.0-flash": 1048576,
            "gemini-1.5-pro": 1048576
        }
        return model_contexts.get(self.model, 32768)  # Default to 32K if unknown
    
    def get_default_model(self):
        """Return the default Gemini model"""
        return "gemini-2.5-pro-exp-03-25"


class OllamaProvider(LLMProvider):
    """Ollama API provider"""
    
    def supports_direct_pdf(self):
        return False

    def setup(self):
        """Initialize Ollama settings"""
        self.model = self.config.get("model", self.get_default_model())
        self.base_url = self.config.get("base_url", "http://localhost:11434")
    
    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=8192):
        """Process document with Ollama API"""
        max_context = self.get_max_context_size()
        
        data = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.get("temperature", 0.2),
                "num_ctx": min(max_context, 24576),
                "num_predict": max_tokens,
                "stop": ["</input>", "</task>"]
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data
        )
        response.raise_for_status()
                        
        response_data = response.json()
        summary_content = response_data.get('response', '')

        if 'error' in response_data:
            raise ValueError(f"Ollama error: {response_data['error']}")
            
        if not summary_content:
            raise ValueError("No content received from Ollama")
            
        return summary_content
    
    def get_max_context_size(self):
        """Return maximum context size for Ollama model"""
        # This would ideally query the Ollama API for the model's context size
        model_contexts = {
            "qwen2.5:14b-instruct-q8_0": 32768, 
            "llama3.1:8b-instruct-q8_0": 131072
        }
        return model_contexts.get(self.model.lower(), 8192)
    
    def get_default_model(self):
        """Return the default Ollama model"""
        return "qwen2.5:14b-instruct-q8_0"


# Factory function to create the appropriate provider
def create_llm_provider(provider_name, config=None):
    """Create an LLM provider instance based on the provider name"""
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
