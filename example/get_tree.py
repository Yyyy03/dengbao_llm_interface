import pickle
import json
from queue import Queue
from raptor.tree_structures import Node,Tree
import os


trees = os.listdir("../demo/raptor-trees/GB")
for tree in trees:
    file_name = tree

    with open(f"../demo/raptor-trees/GB/{file_name}", "rb") as file:
        tree = pickle.load(file)
    data_index=dict()
    root = tree.root_nodes

    q=Queue()
    q.put(root.index)
    print(root.index)

    data={"index":root.index, "text":root.text,"children":[]}
    data_index[root.index]=data

    while not q.empty():
        cur = q.get()
        for child in tree.all_nodes[cur].children:
            q.put(tree.all_nodes[child].index)
            tmp = {"index":tree.all_nodes[child].index,"text":tree.all_nodes[child].text,"children":[]}
            data_index[cur]["children"].append(tmp)
            data_index[tree.all_nodes[child].index] = tmp
    print(data)
    with open(f"../tree-json/{file_name}.json", 'w', encoding="utf8") as f:
        json.dump(data,f,ensure_ascii=False)

# get the node of 3



# for child in root.children:
#     if tree.all_nodes[child].text == ""