# How QC team would use the protected API
import requests
import json
#ws_test_Xy9Zw8Uv7Ts6Rq5Po4Nm3Lk2Jh1Ig
headers = {"X-API-Key": "ws_test_"}

response = requests.post(
    "http://localhost:8000/v1/translate",
    headers=headers,
    files={"json_file": open("C:\\Users\\Gunjan\\Desktop\\gunjan\\Auto_Tag_Correction\\domx-document_transcosmos.json", "rb")},
    data={"language": "Japanese"}
)

# Print status code for debugging
print(f"Status Code: {response.status_code}")

# Handle different response codes
if response.status_code == 200:
    # Success - print the translated JSON
    result = response.json()
    translated_json = json.loads(result.get("translated_json", "[]"))

    print("\n=== Translation Successful ===")
    print(f"Number of translated nodes: {len(translated_json)}")
    print("\nTranslated content:")

    # Pretty print the translated JSON
    for item in translated_json[:5]:  # Show first 5 translations
        print(f"\nNode ID: {item.get('id')}")
        print(f"Original Text: {item.get('text', 'N/A')[:100]}...")  # First 100 chars
        print("-" * 50)

    # Print full JSON if you want to see everything
    print("\nFull translated JSON:")
    print(json.dumps(translated_json, ensure_ascii=False, indent=2))

elif response.status_code == 401:
    print("Error: Missing API key")
elif response.status_code == 403:
    print("Error: Invalid API key or no permission")
elif response.status_code == 429:
    error_detail = response.json().get("detail", "Rate limit exceeded")
    print(f"Error: {error_detail}")
else:
    # Handle any other error
    print(f"Error {response.status_code}: {response.text}")