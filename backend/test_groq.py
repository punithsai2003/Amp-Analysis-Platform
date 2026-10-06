"""
Test Groq API Connection
"""
import requests
import json

# Your API key
GROQ_API_KEY = "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"

print("=" * 60)
print("GROQ API CONNECTION TEST")
print("=" * 60)

# Test 1: List available models
print("\n1️⃣ Testing API Key - Listing Models...")
try:
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        timeout=10
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ API Key is VALID!")
        models = response.json()
        print(f"Available models: {len(models.get('data', []))}")
        for model in models.get('data', [])[:3]:
            print(f"  - {model.get('id')}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Exception: {e}")

# Test 2: Simple chat completion
print("\n2️⃣ Testing Chat Completion...")
try:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": "Say 'Hello, World!' in one sentence."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print(f"URL: https://api.groq.com/openai/v1/chat/completions")
    print(f"Method: POST")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    if response.status_code == 200:
        data = response.json()
        message = data['choices'][0]['message']['content']
        print(f"\n✅ SUCCESS!")
        print(f"AI Response: {message}")
    else:
        print(f"\n❌ ERROR!")
        print(f"Full response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try with Groq SDK
print("\n3️⃣ Testing with Groq SDK...")
try:
    from groq import Groq
    
    client = Groq(api_key=GROQ_API_KEY)
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Say 'Hello from SDK!' in one sentence."}
        ],
        max_tokens=50
    )
    
    print(f"✅ SDK SUCCESS!")
    print(f"AI Response: {completion.choices[0].message.content}")
    
except ImportError:
    print("⚠️ Groq SDK not installed. Run: pip install groq")
except Exception as e:
    print(f"❌ SDK Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)