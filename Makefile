.PHONY: install run engine genai report clean help

help:
	@echo "MetricGuard — AI Metric-Consistency Auditor"
	@echo ""
	@echo "  make install   install Python dependencies"
	@echo "  make run       run the full pipeline (engine → genai → report)"
	@echo "  make engine    run the semantic matching engine only"
	@echo "  make genai     run the RAG + LLM agent layer only"
	@echo "  make report    regenerate output/results.json and output/data.js"
	@echo "  make clean     remove __pycache__ and .pyc files"
	@echo ""
	@echo "Set ANTHROPIC_API_KEY for live LLM recommendations."

install:
	pip3 install -r requirements.txt

engine:
	cd src && python3 engine.py

genai:
	cd src && python3 genai.py

report:
	cd src && python3 report.py
	@echo "Open output/dashboard.html in your browser."

run: engine genai report

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; true
