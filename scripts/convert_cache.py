import ijson
import sqlite3

# Path to your large JSON file
json_file_path = 'cache.json'

# Initialize the SQLite database
db_path = 'model_responses_cache.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table if it does not exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS json_data (
    id INTEGER PRIMARY KEY,
    model_name TEXT,
    prompt TEXT,
    content TEXT,
    stop_reason TEXT
)
''')
conn.commit()

def process_and_store_json_item(item):
    # Assuming each item is a dictionary with keys matching the database columns
    cursor.execute('''
    INSERT INTO json_data (model_name, prompt, content, stop_reason)
    VALUES (?, ?, ?, ?)
    ''', (item['model_name'], item['prompt'], item['content'], item['stop_reason']))
    conn.commit()

# Open the JSON file and incrementally process it
with open(json_file_path, 'rb') as f:  # Use 'rb' mode for binary reading
    items = ijson.items(f, 'item')  # Adjust the prefix according to your JSON structure
    for item in items:
        process_and_store_json_item(item)

conn.close()

