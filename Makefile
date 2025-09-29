.PHONY: install run ui test clean

install:
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
python main.py

ui:
streamlit run ui/app.py

test:
pytest -q

clean:
rm -rf artifacts docs/MODEL_CARD.md docs/MONITORING_DASHBOARD.md
