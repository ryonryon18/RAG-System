import sqlite3
import os

def init_db():
    conn = sqlite3.connect('data/error_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT,
                        prediction TEXT,
                        reference TEXT,
                        error_type TEXT
                      )''')
    conn.commit()
    conn.close()

def save_error(question, prediction, reference, error_type):
    conn = sqlite3.connect('data/error_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO errors (question, prediction, reference, error_type) VALUES (?, ?, ?, ?)",
                   (question, prediction, reference, error_type))
    conn.commit()
    conn.close()

def get_errors():
    conn = sqlite3.connect('data/error_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM errors")
    errors = cursor.fetchall()
    conn.close()
    return errors

if __name__ == "__main__":
    init_db()
