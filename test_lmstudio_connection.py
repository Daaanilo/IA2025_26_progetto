"""Quick test for LM Studio connection"""
import lmstudio as lms

print("Testing LM Studio connection...")
print("Server: http://127.0.0.1:1234")
print(f"lmstudio version: {lms.__version__}")

try:
    # v1.5.0 official syntax: use context manager
    with lms.Client() as client:
        print("✓ Client created!")
        
        # List loaded models (returns LLM objects, not model info)
        models = client.llm.list_loaded()
        print(f"\nLoaded models count: {len(models)}")
        
        # Connect to specific model (must be already loaded in LM Studio)
        # Use the exact model identifier as shown in LM Studio
        model = client.llm.model("llama-3.2-3b-instruct")
        print(f"✓ Model connected: llama-3.2-3b-instruct")
        
        # Test prediction
        print("\nTesting prediction...")
        result = model.respond("Say 'test successful' and nothing else.")
        print(f"✓ Response: {result}")
        
        print("\n" + "="*50)
        print("CONNECTION TEST PASSED!")
        print("="*50)
    
except Exception as e:
    print(f"\n✗ Connection failed: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    
    print("\nTroubleshooting:")
    print("1. Ensure LM Studio is running")
    print("2. Go to 'Developer' tab in LM Studio")
    print("3. Click 'Start Server' (should show port 1234)")
    print("4. Load 'llama-3.2-3b-instruct' model")