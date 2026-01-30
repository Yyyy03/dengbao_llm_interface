import markdown
import re
import config as cf
import requests
from raptor.utils import image_to_decp


def parse_markdown_to_tree(markdown_text):
    # 正则表达式匹配 Markdown 标题
    header_pattern = re.compile(r'^(#{1,6})\s*(.*)$', re.MULTILINE)
    # 正则表达式匹配非标题内容
    content_pattern = re.compile(r'^[^#].*$', re.MULTILINE)

    # 提取所有标题和内容
    headers = header_pattern.findall(markdown_text)
    contents = content_pattern.findall(markdown_text)

    # 初始化树形结构
    tree = {}
    stack = []  # 用于存储当前层级的节点路径

    # 当前处理到的 Markdown 内容的索引
    content_index = 0

    for i,header in enumerate(headers):
        level = len(header[0])  # 标题层级
        title = header[1].strip()  # 标题文本

        # 如果当前标题层级小于栈顶层级，需要回退到合适的层级
        while stack and stack[-1][0] >= level:
            stack.pop()

        # 创建当前标题的节点
        node = {"title": title, "children": [], "content": []}

        # 如果栈不为空，将当前节点添加到栈顶节点的子节点中
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            # 否则，当前节点是根节点
            tree[title] = node

        # 将当前节点及其层级压入栈
        stack.append((level, node))

        # 处理当前标题下的内容
        first_index = markdown_text.find(f'{header[0]} {header[1]}')
        if i!=len(headers)-1:
            next_index = markdown_text.find(f'{headers[i+1][0]} {headers[i+1][1]}')
        else:
            next_index = len(headers)-1
        if first_index!=-1 and next_index!=-1:
            text = markdown_text[first_index+len(f'{header[0]} {header[1]}'):next_index]
            node["content"] = text


    return tree

def process_image_md(markdown_file,output_dir="./out.md"):
    with open(markdown_file,'r',encoding="utf8") as f:
        content = f.read()
    # content=re.sub(r'\n+','\n',content)
    img_re = re.compile(r'!\[.*?\]\((.*?)\)|<img.*?src=["\'](.*?)["\']', re.I)
    urls = [m[0] or m[1] for m in img_re.findall(content)]

    for url in urls:
        if not url.startswith(("http://", "https://")):
            continue
        try:
            res = requests.get(url)
            replacement = image_to_decp(res.content)
            print(replacement)
            # 把原完整 ![]() 或 <img ...> 整块换掉
            content = re.sub(
                re.escape(f"![]({url})") if f"![]" in content
                else rf'<img[^>]*?src=["\']{re.escape(url)}["\'][^>]*>',
                replacement, content, flags=re.I
            )
            print(f"[INFO] 已替换 {url[:60]}")
        except Exception as e:
            print(f"[WARN] 处理 {url} 失败：{e}")
    with open(output_dir,'w',encoding='utf8') as f:
        f.write(content)

    #
    # if write_back:
    #     md.write_text(content, encoding="utf8")
    return content


if __name__ == '__main__':
    # with open("./22239-new(1).md",'r',encoding="utf8") as f:
    #     content = f.read()
    #
    # content=re.sub(r'\n+','\n',content)
    # tree = parse_markdown_to_tree(content)
    # # print(content)
    # import json
    # with open("../tree.json", "w", encoding="utf8") as f:
    #     json.dump(tree,f,ensure_ascii=False)
    import os
    dirs = os.listdir("../demo/markdowns/other")
    for dir in dirs:
        process_image_md(f"../demo/markdowns/other/{dir}",f"../demo/markdowns/qt/{dir}")

