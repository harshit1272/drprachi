import google.generativeai as genai
import os

# 1. Setup the key explicitly
os.environ["GOOGLE_API_KEY"] = "AIzaSyDBOpBytzB1oihKZBXOOIvIIhVxUHAoKz0"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("------------------------------------------------")
print("📡 Connecting to Google to check your access...")
print("------------------------------------------------")

try:
    # 2. List all available models
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ FOUND: {m.name}")
            available_models.append(m.name)

    if not available_models:
        print("❌ No models found. Your API Key might be invalid or has no project permissions.")
    else:
        print("------------------------------------------------")
        print(f"🎉 Success! You can use: {available_models[0]}")
        print("------------------------------------------------")

except Exception as e:
    print(f"❌ CONNECTION ERROR: {e}")