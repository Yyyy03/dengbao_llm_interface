from raptor.new_tree_builder import NewTreeBuilder

if __name__ == "__main__":
    import json
    with open('../tree.json', 'r', encoding="utf8") as f:
        text = json.load(f)
    a = NewTreeBuilder(None,text)
    nodes = a.construct_tree(None,None,None)
    print(nodes)

