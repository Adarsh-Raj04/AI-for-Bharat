import json

data = {}

with open("backend/.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            data[k] = v

print(json.dumps(data, indent=2))
