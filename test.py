import boto3
import json

# Create Bedrock Runtime client
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"   # Claude works reliably here
)

# Claude model ID (Haiku)
MODEL_ID = "amazon.titan-embed-text-v2:0"

# Request body (Anthropic format for Bedrock)
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 50,
    "messages": [
        {
            "role": "user",
            "content": "Say hi"
        }
    ]
}

# Invoke model
response = client.invoke_model(
    modelId=MODEL_ID,
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json"
)

# Parse response
response_body = json.loads(response["body"].read())

print(response_body["content"][0]["text"])
