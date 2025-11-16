.PHONY: help install clean run-poland run-american run-taiwan run-comparison run-all report paper

# Colors
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m

PYTHON := .venv/bin/python
UV := uv

help:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN) Bankruptcy Prediction - Multi-Dataset Analysis$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make install        - Create .venv (Python 3.13) and install dependencies"
	@echo "  make clean          - Remove generated files"
	@echo ""
	@echo "$(YELLOW)Analysis - Polish Dataset:$(NC)"
	@echo "  make run-poland     - Run complete Polish dataset analysis (scripts 01-07)"
	@echo ""
	@echo "$(YELLOW)Analysis - American Dataset:$(NC)"
	@echo "  make run-american   - Run complete US dataset analysis"
	@echo ""
	@echo "$(YELLOW)Analysis - Taiwan Dataset:$(NC)"
	@echo "  make run-taiwan     - Run complete Taiwan dataset analysis"
	@echo ""
	@echo "$(YELLOW)Cross-Dataset:$(NC)"
	@echo "  make run-comparison - Run cross-dataset comparison"
	@echo "  make run-all        - Run ALL analyses (Poland + US + Taiwan + Comparison)"
	@echo ""
	@echo "$(YELLOW)Reports:$(NC)"
	@echo "  make report         - Generate master HTML report"
	@echo "  make paper          - Compile LaTeX seminar paper (German)"
	@echo ""

install:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN) Installing Dependencies (Python 3.13)$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@if [ ! -d ".venv" ]; then \
		echo "$(BLUE)Creating virtual environment with Python 3.13...$(NC)"; \
		$(UV) venv --python 3.13; \
	fi
	@echo "$(BLUE)Installing dependencies with uv...$(NC)"
	$(UV) sync
	@echo "$(GREEN)✓ Installation complete!$(NC)"
	@echo "$(BLUE)Verifying Python version...$(NC)"
	@$(PYTHON) --version
	@echo ""

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	rm -rf results/script_outputs/*
	rm -rf results/models/*.pkl
	rm -rf data/processed/splits/*
	rm -rf __pycache__ src/**/__pycache__ scripts_python/__pycache__
	find . -name "*.pyc" -delete
	@echo "$(GREEN)✓ Clean complete!$(NC)"

run-poland:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN) Running Polish Dataset Analysis$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)[1/7] Data Understanding...$(NC)"
	@$(PYTHON) scripts_python/01_data_understanding.py
	@echo ""
	@echo "$(YELLOW)[2/7] Exploratory Analysis...$(NC)"
	@$(PYTHON) scripts_python/02_exploratory_analysis.py
	@echo ""
	@echo "$(YELLOW)[3/7] Data Preparation...$(NC)"
	@$(PYTHON) scripts_python/03_data_preparation.py
	@echo ""
	@echo "$(YELLOW)[4/7] Baseline Models...$(NC)"
	@$(PYTHON) scripts_python/04_baseline_models.py
	@echo ""
	@echo "$(YELLOW)[5/7] Advanced Models...$(NC)"
	@$(PYTHON) scripts_python/05_advanced_models.py
	@echo ""
	@echo "$(YELLOW)[6/7] Model Calibration...$(NC)"
	@$(PYTHON) scripts_python/06_model_calibration.py
	@echo ""
	@echo "$(YELLOW)[7/7] Cross-Horizon Robustness...$(NC)"
	@$(PYTHON) scripts_python/07_robustness_analysis.py
	@echo ""
	@echo "$(GREEN)✓ Polish dataset analysis complete!$(NC)"

run-american:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN) Running American Dataset Analysis$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)[1/3] Data Cleaning...$(NC)"
	@$(PYTHON) scripts_python/american/01_data_cleaning.py
	@echo ""
	@echo "$(YELLOW)[2/3] Exploratory Analysis...$(NC)"
	@$(PYTHON) scripts_python/american/02_eda.py
	@echo ""
	@echo "$(YELLOW)[3/3] Baseline Models...$(NC)"
	@$(PYTHON) scripts_python/american/03_baseline_models.py
	@echo ""
	@echo "$(GREEN)✓ American dataset analysis complete!$(NC)"

run-taiwan:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN) Running Taiwan Dataset Analysis$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)[1/3] Data Cleaning...$(NC)"
	@$(PYTHON) scripts_python/taiwan/01_data_cleaning.py
	@echo ""
	@echo "$(YELLOW)[2/3] Exploratory Analysis...$(NC)"
	@$(PYTHON) scripts_python/taiwan/02_eda.py
	@echo ""
	@echo "$(YELLOW)[3/3] Baseline Models...$(NC)"
	@$(PYTHON) scripts_python/taiwan/03_baseline_models.py
	@echo ""
	@echo "$(GREEN)✓ Taiwan dataset analysis complete!$(NC)"

run-comparison:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN) Running Cross-Dataset Comparison$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@$(PYTHON) scripts_python/cross_dataset_comparison.py
	@echo ""
	@echo "$(GREEN)✓ Cross-dataset comparison complete!$(NC)"

run-all:
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN) Running COMPLETE Multi-Dataset Analysis$(NC)"
	@echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@$(MAKE) run-poland
	@echo ""
	@$(MAKE) run-american
	@echo ""
	@$(MAKE) run-taiwan
	@echo ""
	@$(MAKE) run-comparison
	@echo ""
	@$(MAKE) report
	@echo ""
	@echo "$(GREEN)✓✓✓ COMPLETE ANALYSIS FINISHED ✓✓✓$(NC)"

report:
	@echo "$(BLUE)Generating master HTML report...$(NC)"
	@$(PYTHON) scripts_python/generate_html_report.py
	@echo "$(GREEN)✓ Report generated: results/ANALYSIS_REPORT.html$(NC)"

paper:
	@echo "$(BLUE)Compiling LaTeX seminar paper...$(NC)"
	@cd seminar-paper && pdflatex -interaction=nonstopmode doku_main.tex
	@cd seminar-paper && biber doku_main
	@cd seminar-paper && pdflatex -interaction=nonstopmode doku_main.tex
	@cd seminar-paper && pdflatex -interaction=nonstopmode doku_main.tex
	@echo "$(GREEN)✓ Paper compiled: seminar-paper/doku_main.pdf$(NC)"
