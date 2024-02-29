import sqlite3
from dataclasses import dataclass


@dataclass
class ModelResponse:
    model_name: str
    content: str
    stop_reason: str


class ModelResponseCache:
    def __init__(self, db_path='model_responses_cache.db'):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            prompt TEXT,
            content TEXT,
            stop_reason TEXT,
            UNIQUE(model_name, prompt)
        )''')
        conn.commit()
        conn.close()

    def cache_response(self, model_name, prompt, response):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
            INSERT OR REPLACE INTO responses (model_name, prompt, content, stop_reason)
            VALUES (?, ?, ?, ?)
            ''', (model_name, prompt, response.content, response.stop_reason))
            conn.commit()

    def get_cached_response(self, model_name, prompt):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
            SELECT content, stop_reason FROM responses WHERE model_name = ? AND prompt = ?
            ''', (model_name, prompt))
            result = c.fetchone()
            if result:
                return ModelResponse(model_name, result[0], result[1])

    def is_key_stored(self, model_name, prompt):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
            SELECT EXISTS(SELECT 1 FROM responses WHERE model_name = ? AND prompt = ? LIMIT 1)
            ''', (model_name, prompt))
            return c.fetchone()[0] == 1