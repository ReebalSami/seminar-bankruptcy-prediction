.PHONY: help install clean run-poland run-american run-taiwan run-comparison run-all report paper \
        phase01 phase02 phase03 phase04 \
        phase04-tables phase05-modeling phase05-eval phase05-tables phase05-modeling-extra \
        phase04d-consensus-nested phase05-modeling-nested phase05-modeling-extra-nested phase06-eval-nested \
        delta-addendum addendum

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
	@echo "$(YELLOW)Pipeline Phases:$(NC)"
	@echo "  make phase01        - Data preparation (duplicates, outliers, imputation)"
	@echo "  make phase02        - Exploratory analysis (distributions, tests, correlations)"
	@echo "  make phase03        - Multicollinearity analysis (VIF)"
	@echo "  make phase04        - Feature selection (filter, wrapper, embedded, consensus)"
	@echo "  make phase05-modeling - Base models (LR, RF)"
	@echo "  make phase05-modeling-extra - Extra models + ensemble"
	@echo "  make phase05-eval   - Aggregate metrics and plots"
	@echo ""
	@echo "$(YELLOW)Paper Tables:$(NC)"
	@echo "  make phase04-tables - Generate Phase 04 LaTeX tables for paper"
	@echo "  make phase05-tables - Generate Phase 05 LaTeX tables for paper"
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

# -----------------------------------------------------------------------------
# Complete Pipeline Phases
# -----------------------------------------------------------------------------
phase01:
	@echo "$(BLUE)Phase 01: Data Preparation...$(NC)"
	@$(PYTHON) scripts/01_data_preparation/01a_remove_duplicates.py
	@$(PYTHON) scripts/01_data_preparation/01b_outlier_treatment.py
	@$(PYTHON) scripts/01_data_preparation/01c_missing_value_imputation.py
	@$(PYTHON) scripts/01_data_preparation/01d_create_horizon_datasets.py
	@echo "$(GREEN)✓ Phase 01 complete: data/processed/poland_imputed.parquet$(NC)"

phase02:
	@echo "$(BLUE)Phase 02: Exploratory Analysis...$(NC)"
	@$(PYTHON) scripts/02_exploratory_analysis/02a_distribution_analysis.py
	@$(PYTHON) scripts/02_exploratory_analysis/02b_univariate_tests.py
	@$(PYTHON) scripts/02_exploratory_analysis/02c_correlation_economic.py
	@echo "$(GREEN)✓ Phase 02 complete: results/02_exploratory_analysis/$(NC)"

phase03:
	@echo "$(BLUE)Phase 03: Multicollinearity Analysis...$(NC)"
	@$(PYTHON) scripts/03_multicollinearity/03a_vif_analysis.py
	@echo "$(GREEN)✓ Phase 03 complete: results/03_multicollinearity/$(NC)"

phase04:
	@echo "$(BLUE)Phase 04: Feature Selection...$(NC)"
	@$(PYTHON) scripts/04_feature_selection/04a_filter_methods.py
	@$(PYTHON) scripts/04_feature_selection/04b_wrapper_methods.py
	@$(PYTHON) scripts/04_feature_selection/04c_embedded_methods.py
	@$(PYTHON) scripts/04_feature_selection/04d_stability_consensus.py
	@echo "$(GREEN)✓ Phase 04 complete: data/processed/feature_sets_selected/$(NC)"

phase04-tables:
	@echo "$(BLUE)Generating Phase 04 LaTeX tables...$(NC)"
	@$(PYTHON) scripts/paper_helper/generate_phase04_tables_v2.py
	@echo "$(GREEN)✓ Tables updated under seminar-paper/tables$(NC)"

phase05-modeling:
	@echo "$(BLUE)Running Phase 05 Modeling (LR + RF)...$(NC)"
	@$(PYTHON) scripts/05_modeling/05_modeling_train_evaluate.py
	@echo "$(GREEN)✓ Modeling outputs in results/05_modeling$(NC)"

phase05-modeling-extra:
	@echo "$(BLUE)Running Phase 05 Extra Models (GB, ET, SVC, Ensemble)...$(NC)"
	@$(PYTHON) scripts/05_modeling/05b_modeling_extra_models.py
	@echo "$(GREEN)✓ Extra modeling outputs in results/05_modeling/extra$(NC)"

phase05-eval:
	@echo "$(BLUE)Aggregating Phase 05 model metrics...$(NC)"
	@$(PYTHON) scripts/06_model_evaluation/06_aggregate_and_plots.py
	@echo "$(GREEN)✓ Evaluation outputs in results/06_model_evaluation$(NC)"

phase05-tables:
	@echo "$(BLUE)Generating Phase 05 LaTeX tables...$(NC)"
	@$(PYTHON) scripts/paper_helper/generate_phase05_tables.py
	@$(PYTHON) scripts/paper_helper/generate_phase05_comparison_tables.py
	@$(PYTHON) scripts/paper_helper/generate_phase06_tables.py
	@echo "$(GREEN)✓ Tables updated under seminar-paper/tables$(NC)"

paper-assets:
	@echo "$(BLUE)Copying paper assets...$(NC)"
	@$(UV) run python scripts/paper_helper/copy_phase00_assets.py
	@$(UV) run python scripts/paper_helper/copy_phase01_assets.py
	@$(UV) run python scripts/paper_helper/copy_phase02_assets.py
	@$(UV) run python scripts/paper_helper/generate_phase03_assets.py
	@$(UV) run python scripts/paper_helper/copy_phase03_assets.py
	@$(UV) run python scripts/paper_helper/generate_phase04_assets.py
	@$(UV) run python scripts/paper_helper/copy_phase04_assets.py
	@$(UV) run python scripts/paper_helper/generate_phase05_assets.py
	@$(UV) run python scripts/paper_helper/copy_phase05_assets.py
	@$(UV) run python scripts/paper_helper/generate_phase03_tables.py
	@$(UV) run python scripts/paper_helper/generate_phase05_tables.py
	@$(UV) run python scripts/paper_helper/generate_phase05_comparison_tables.py
	@$(UV) run python scripts/paper_helper/generate_phase06_tables.py
	@echo "$(GREEN)✓ Paper assets copied$(NC)"

# -----------------------------------------------------------------------------
# Nested variant pipeline (v1.1): write to *_nested dirs and suffixed files
# -----------------------------------------------------------------------------
phase04d-consensus-nested:
	@echo "$(BLUE)Phase 04d (nested): Stability & Consensus...$(NC)"
	@$(PYTHON) scripts/04_feature_selection/04d_stability_consensus.py --variant nested
	@echo "$(GREEN)✓ 04d nested outputs ready$(NC)"

phase05-modeling-nested:
	@echo "$(BLUE)Phase 05 (nested): Baseline LR + RF...$(NC)"
	@$(PYTHON) scripts/05_modeling/05_modeling_train_evaluate.py --variant nested
	@echo "$(GREEN)✓ 05_modeling_nested complete$(NC)"

phase05-modeling-extra-nested:
	@echo "$(BLUE)Phase 05b (nested): Extra models + Ensemble...$(NC)"
	@$(PYTHON) scripts/05_modeling/05b_modeling_extra_models.py --variant nested
	@echo "$(GREEN)✓ 05_modeling_nested/extra complete$(NC)"

phase06-eval-nested:
	@echo "$(BLUE)Phase 06 (nested): Aggregate modeling metrics...$(NC)"
	@$(PYTHON) scripts/06_model_evaluation/06_aggregate_and_plots.py --variant nested
	@echo "$(GREEN)✓ 06_model_evaluation_nested complete$(NC)"

delta-addendum:
	@echo "$(BLUE)Generating delta assets (v1.0 → v1.1)...$(NC)"
	@$(PYTHON) scripts/paper_helper/generate_addendum_assets.py
	@$(PYTHON) scripts/paper_helper/generate_addendum_text.py
	@echo "$(GREEN)✓ Delta assets generated$(NC)"

addendum:
	@echo "$(BLUE)Compiling Post-Submission Addendum...$(NC)"
	@cd seminar-paper/addendum && pdflatex -interaction=nonstopmode addendum.tex
	@cd seminar-paper/addendum && pdflatex -interaction=nonstopmode addendum.tex
	@echo "$(GREEN)✓ Addendum compiled: seminar-paper/addendum/addendum.pdf$(NC)"
