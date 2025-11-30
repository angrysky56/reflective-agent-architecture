import logging
import re

import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockConfig:
    llm_model = "deepseek-r1:1.5b"

class MockWorkspace:
    def __init__(self):
        self.config = MockConfig()

    def _llm_generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 8000) -> str:
        """
        Generate text using local Ollama LLM.
        """
        try:
            # Fast-path: avoid implicit model pull by verifying availability first
            try:
                _ = ollama.show(model=self.config.llm_model)
            except Exception as avail_err:
                logger.warning(f"LLM model '{self.config.llm_model}' not available: {avail_err}")
                raise RuntimeError("LLM model unavailable")

            # Enhanced system prompt with framework context
            enhanced_system = f"""You are a text generation component in a cognitive reasoning system.

Your role: Generate concise, clear text outputs. The system handles reasoning via graph and vector operations.

Framework: Cognitive Workspace Database (System 2 Reasoning)
- Neo4j: Structural/graph reasoning
- Chroma: Vector similarity in latent space
- Your task: Bridge text generation only

{system_prompt}

CRITICAL: Output your final answer directly. You may think internally, but end with clear, concise output."""

            print(f"Sending request to model: {self.config.llm_model}")
            response = ollama.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": user_prompt},
                ],
                options={"num_predict": max_tokens, "temperature": 0.7},
            )
            content = response["message"]["content"].strip()
            print(f"Raw LLM output: {repr(content)}")

            # Strip reasoning artifacts that models add
            # Remove <think>...</think> blocks, but be careful not to delete everything
            if "<think>" in content:
                # Try to extract content after </think>
                parts = re.split(r"</think>", content, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[-1].strip():
                    content = parts[-1].strip()
                else:
                    # If everything is inside <think> or no closing tag, just strip the tags
                    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)

            # Remove common reasoning prefixes (case-insensitive, at start of content)
            reasoning_patterns = [
                r"^(?:Okay|Alright|Let me|Hmm|So|Well|First|Now)\s*[,:]?\s*",
                r"^(?:The user|I need to|I should|Looking at)\s+.*?\.\s*",
            ]

            for pattern in reasoning_patterns:
                # Only replace if it leaves something behind
                if re.match(pattern, content, flags=re.IGNORECASE):
                    new_content = re.sub(pattern, "", content, count=1, flags=re.IGNORECASE).strip()
                    if new_content:
                        content = new_content

            # Extract actual content after reasoning markers
            # Look for explicit markers like "OUTPUT:"
            output_markers = [
                r"(?:OUTPUT|ANSWER|RESULT|FINAL):\s*(.+)",  # Explicit markers
            ]

            for marker_pattern in output_markers:
                match = re.search(marker_pattern, content, re.DOTALL | re.IGNORECASE)
                if match and len(match.group(1).strip()) > 20:
                    content = match.group(1).strip()
                    break

            # Clean up multiple spaces and newlines
            content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)  # Max 2 newlines
            content = re.sub(r"[ \t]+", " ", content)  # Normalize spaces
            content = content.strip()

            if not content and response["message"]["content"].strip():
                # If stripping removed everything, revert to raw content (safety net)
                logger.warning("Stripping removed all content, reverting to raw output")
                content = response["message"]["content"].strip()

            return content if content else "[No output generated]"
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return f"[LLM unavailable: {user_prompt[:50]}...]"

# Test with a simple prompt
workspace = MockWorkspace()
print("\n--- Running Simple Test ---")
result = workspace._llm_generate("You are a helpful assistant.", "Say hello.", max_tokens=100)
print(f"\nResult: {repr(result)}")
