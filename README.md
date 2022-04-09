## Setup

1. Place user_behaviour.json in the data subdirectory.

2. Install dependencies into a virtual env:

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

## How to run

- Create the baseline model and produce averaged performance plots:

`python nifty/recommend.py`

- Produce performance plots for a given user by ID:

`python nifty/active_user.py 02e383e1-5558-4267-bbf5-27c1cf255acb`
