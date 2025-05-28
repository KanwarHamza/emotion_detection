# ---------- core/mmse_tasks.py ----------
MMSE_TASKS = {
    "orientation": [
        {"question": "What year is it?", "answer": "2024", "points": 1},
        {"question": "Current season?", "answer": "summer", "points": 1}
    ],
    "memory": [
        {"question": "Remember: Apple, Table, Penny", "answer": ["apple", "table", "penny"], "points": 3}
    ],
    "attention": [
        {"question": "Count down from 20 by 3", "answer": [20, 17, 14, 11, 8, 5, 2], "points": 5}
    ],
    "language": [
        {"question": "Repeat: 'No ifs, ands, or buts'", "answer": "no ifs ands or buts", "points": 1}
    ]
}
