# Sphinx 文档构建的最小 Makefile
#

# 以下变量可以通过命令行设置
# 前两个变量也可以通过环境变量设置
SPHINXOPTS    ?=    # Sphinx 构建选项
SPHINXBUILD   ?= sphinx-build    # Sphinx 构建命令
SOURCEDIR     = .    # 源文件目录
BUILDDIR      = build    # 构建输出目录

# 将 help 目标放在第一位，这样直接运行 "make" 命令时就相当于运行 "make help"
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 声明 help 和 Makefile 为伪目标
.PHONY: help Makefile

# 通配符目标：将所有未知目标都路由到 Sphinx
# 使用新的 "make mode" 选项。$(O) 是 $(SPHINXOPTS) 的快捷方式
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
