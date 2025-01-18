# Sphinx 文档构建器的配置文件
#
# 此文件仅包含最常用的选项。完整的选项列表请参见：
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- 路径设置 --------------------------------------------------------------

# 如果扩展（或要使用 autodoc 记录的模块）在另一个目录中，
# 请在此处将这些目录添加到 sys.path。如果目录相对于文档根目录，
# 请使用 os.path.abspath 将其转换为绝对路径，如下所示。
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- 项目信息 -----------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import pettingzoo  # noqa: E402

project = "PettingZoo"
copyright = "2023 Farama Foundation"
author = "Farama Foundation"

# 完整版本号，包括 alpha/beta/rc 标签
release = pettingzoo.__version__


# -- 通用配置 ---------------------------------------------------

# 在此处添加 Sphinx 扩展模块名称（字符串形式）。
# 可以是 Sphinx 自带的扩展（名为 'sphinx.ext.*'）
# 或者是自定义扩展。
extensions = [
    "sphinx.ext.napoleon",     # 支持 NumPy 和 Google 风格的文档字符串
    "sphinx.ext.doctest",      # 文档测试支持
    "sphinx.ext.autodoc",      # 自动文档生成
    "sphinx.ext.githubpages",  # GitHub Pages 支持
    "sphinx.ext.intersphinx",  # 跨文档引用支持
    "sphinx.ext.viewcode",     # 查看源代码链接
    "myst_parser",             # Markdown 支持
    "sphinx_github_changelog", # GitHub 更新日志支持
]

# 添加包含模板的路径（相对于此目录）
templates_path = ["_templates"]

# 搜索源文件时要忽略的文件和目录的模式列表（相对于源目录）
# 此模式也会影响 html_static_path 和 html_extra_path
exclude_patterns = []

# Napoleon 设置
napoleon_use_ivar = True  # 使用 :ivar: 指令
napoleon_use_admonition_for_references = True  # 为引用使用警告框
# 参见 https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]  # 自定义章节样式

# -- Autodoc 选项 -------------------------------------------------

autoclass_content = "both"  # 包含类和构造函数文档
autodoc_preserve_defaults = True  # 保留默认值

# -- Intersphinx 选项 -----------------------------------------------

# 外部项目文档映射
intersphinx_mapping = {
    "shimmy": ("https://shimmy.farama.org/", None),
}
intersphinx_disabled_reftypes = ["*"]  # 禁用所有引用类型

# -- HTML 输出选项 -------------------------------------------------

# HTML 和 HTML Help 页面使用的主题。
# 查看可用的内置主题列表。
#
html_theme = "furo"  # 使用 Furo 主题
html_title = "PettingZoo 文档"
html_baseurl = "https://pettingzoo.farama.org"
html_copy_source = False  # 不复制源文件
html_favicon = "_static/img/favicon.png"  # 网站图标
html_theme_options = {
    "light_logo": "img/PettingZoo.svg",        # 亮色主题 logo
    "dark_logo": "img/PettingZoo_White.svg",   # 暗色主题 logo
    "gtag": "G-Q4EGMJ3R24",                    # Google Analytics 标签
    "versioning": True,                         # 启用版本控制
    "source_repository": "https://github.com/Farama-Foundation/PettingZoo/",  # 源代码仓库
    "source_branch": "master",                  # 源代码分支
    "source_directory": "docs/",                # 文档目录
}

# 静态文件路径
html_static_path = ["_static"]
html_css_files = []

# -- MyST 解析器选项 -------------------------------------------------

myst_heading_anchors = 3  # 标题锚点深度

# -- 生成更新日志 -------------------------------------------------

# GitHub 更新日志生成令牌
sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
