

# ——— 变量定义 ———
MANAGER        := conda
ENV_NAME       := tabriskscore
PYTHON_VERSION := 3.9      

# 要进行格式化 / 检查的源码目录
PYTHON_FILES   := src/tabriskscore

# 在该 Conda 环境里运行命令的前缀
RUN_CMD        := $(MANAGER) run -n $(ENV_NAME)

# ——— 任务定义 ———

.PHONY: create-env
create-env:
	@echo "==> Creating conda environment '$(ENV_NAME)' (Python $(PYTHON_VERSION))"
	$(MANAGER) create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

.PHONY: install-pkg
install-pkg:
	@echo "==> Installing TabRiskScore + QBS (editable mode) into '$(ENV_NAME)'"
	# 1) 安装主包 tabriskscore
	$(RUN_CMD) pip install -e .
	# 2) 安装 QBS/OptimQBS：让 querysnout/src/optimized_qbs 里的 setup.py 生效
	cd querysnout/src/optimized_qbs && $(RUN_CMD) pip install -e .

.PHONY: install-old-mbi
install-old-mbi:
	@echo "==> Installing old-version private-pgm into '$(ENV_NAME)'"
	$(RUN_CMD) pip install git+https://github.com/ryan112358/private-pgm.git@4152cc591f30560ce994870318ee4801b016bf94

.PHONY: install-kernel
install-kernel:
	@echo "==> Installing ipykernel in '$(ENV_NAME)'"
	$(RUN_CMD) pip install ipykernel
	$(RUN_CMD) python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python($(ENV_NAME))"

.PHONY: setup-env
setup-env:
	@echo "==> Running full setup: create-env → install-pkg → install-old-mbi"
	$(MAKE) create-env
	$(MAKE) install-pkg
	$(MAKE) install-old-mbi
	@echo "✅ Environment '$(ENV_NAME)' is ready!  Use 'conda activate $(ENV_NAME)' to activate."

.PHONY: py-format
py-format:
	@echo "==> Formatting python code with ruff in $(PYTHON_FILES)"
	$(RUN_CMD) ruff format $(PYTHON_FILES)

.PHONY: py-lint
py-lint:
	@echo "==> Linting (ruff check) python code in $(PYTHON_FILES)"
	$(RUN_CMD) ruff check --fix $(PYTHON_FILES)
