.PHONY: install run ui test clean

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	python main.py

ui:
	streamlit run ui/app.py

test:
	mkdir -p test_results
	pytest -v --tb=short --junitxml=test_results/junit.xml --html=test_results/report.html --self-contained-html

clean:
	rm -rf artifacts docs/MODEL_CARD.md docs/MONITORING_DASHBOARD.md test_results
