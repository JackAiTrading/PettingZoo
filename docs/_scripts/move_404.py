"""修复 404 页面中的相对路径链接。

这个脚本将 404 页面中的相对路径链接（以 '../' 开头）转换为绝对路径链接（以 '/' 开头）。
这样可以确保 404 页面中的链接在任何位置都能正常工作。
"""

import sys

if __name__ == "__main__":
    # 检查是否提供了文件路径参数
    if len(sys.argv) < 2:
        print("请提供文件路径")
    filePath = sys.argv[1]

    # 读取文件内容，替换路径，然后写回文件
    with open(filePath, "r+") as fp:
        content = fp.read()
        # 将 '../' 开头的相对路径替换为 '/' 开头的绝对路径
        content = content.replace('href="../', 'href="/').replace('src="../', 'src="/')
        # 清空文件并写入新内容
        fp.seek(0)
        fp.truncate()

        fp.write(content)
