"""Base provider class for the Science Paper Summariser.

Defines the interface that all providers (API and CLI) must implement.
"""


class Provider:
    """Base class for all LLM providers.

    Subclasses must implement:
        setup()              — Initialise provider-specific clients or validate tools.
        process_document()   — Send content to the LLM and return the summary text.
        get_max_context_size() — Return the context window size for this provider family.

    Optionally override:
        supports_direct_pdf()    — Whether the provider accepts raw PDF bytes (default: False).
        validate_runtime_ready() — Pre-run connectivity/model check (default: no-op).
        get_preferred_max_tokens() — Max output tokens for this provider (default: 12 288).
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.model = self.config.get("model")
        self.setup()

    def setup(self):
        """Initialise provider-specific components. Override in subclasses."""
        pass

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        """Process a document and return the generated summary text.

        Args:
            content: Document content — bytes for direct PDF upload, str for extracted text.
            is_pdf: Whether the source file is a PDF.
            system_prompt: System prompt with role, rules, and knowledge base.
            user_prompt: User prompt with task, template, and optionally paper text.
            max_tokens: Maximum tokens for the response generation.

        Returns:
            str: The generated summary text.
        """
        raise NotImplementedError

    def get_max_context_size(self):
        """Return maximum context size for this provider family."""
        raise NotImplementedError

    def supports_direct_pdf(self):
        """Return whether this provider can process PDFs directly without text extraction."""
        return False

    def validate_runtime_ready(self):
        """Assert the provider is reachable and ready before the processing loop starts.

        The default implementation is a no-op. Local/self-hosted providers should
        override this to verify endpoint connectivity and model availability, raising
        ValueError with a clear message if the check fails.
        """
        pass

    def get_preferred_max_tokens(self):
        """Return the preferred max output tokens for this provider.

        Reads from config key 'max_output_tokens' when set; otherwise falls back to
        12 288, which suits frontier API models. Local providers override the default
        to something more generous (32 768) to accommodate thinking tokens and avoid
        silent truncation of long summaries.
        """
        return int(self.config.get("max_output_tokens", 12288))
