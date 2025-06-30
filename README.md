# luna1-memory
Persistent memory layer for Luna #1


# Run server

source myenv/bin/activate
chroma run --port 1111 --path ./data
python main.py

# Test
pytest test.py -q
