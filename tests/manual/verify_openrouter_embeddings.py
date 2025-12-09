import os
import sys

from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.embeddings.embedding_factory import EmbeddingFactory


def verify_openrouter():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment.")
        print("Please set it in .env to run this verification.")
        return

    print("🚀 Initializing OpenRouter Embedding Provider...")
    try:
        # Force provider to be openrouter for this test
        provider = EmbeddingFactory.create(
            provider_name="openrouter", model_name="openai/text-embedding-3-small"
        )

        text = "The quick brown fox jumps over the lazy dog."
        print(f"📝 Encoding text: '{text}'")

        embedding = provider.encode(text)

        dim = provider.get_sentence_embedding_dimension()
        print(f"✅ Success! Embedding generated.")
        print(f"📏 Dimension: {dim}")
        print(f"🔢 First 5 values: {embedding[:5]}")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_openrouter()
