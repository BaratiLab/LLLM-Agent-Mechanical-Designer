import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from truss_y import *
import seaborn as sns
import pandas as pd
import json
import ast

def plot_truss(t):

    plt.figure()
    ax = plt.gca()
    width = 0.35
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(width)
    tick_width = 0.35
    plt.tick_params(direction = 'in', width = tick_width)

    cmap = plt.cm.jet
    nodes_dict = {node[0]: (node[1], node[2]) for node in t.nodes}

    for node in t.nodes:
        plt.plot(node[1],node[2], 'b', marker=".", markersize=10)

    for member in t.members:
        nodes = t.members[member]
        thiccc= t._area[member]/1.25
        plt.plot([nodes_dict[nodes[0]][0], nodes_dict[nodes[1]][0]], [nodes_dict[nodes[0]][1], nodes_dict[nodes[1]][1]], color = "black", linewidth=thiccc)

    for supp in t.supports:
        if t.supports[supp] == "pinned":
            plt.plot(nodes_dict[supp][0], nodes_dict[supp][1], 'r', marker='s', markersize=10)
        if t.supports[supp] == "roller":
            plt.plot(nodes_dict[supp][0], nodes_dict[supp][1], 'g', marker='^', markersize=10)

    for node, load_list in t.loads.items():
        for load in load_list:
            magnitude, direction = (load)
            x,y = nodes_dict[node]
            # print(magnitude, direction, x, y)
            if not str(magnitude).startswith("R"):            
                angle = np.radians(float(direction))
                dx = ((magnitude)/abs(magnitude)) * np.cos(float(angle))
                dy = ((magnitude)/abs(magnitude)) * np.sin(float(angle))
                plt.arrow(x, y, float(dx)/2, float(dy)/2, head_width=0.1)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)


    stress = t.member_stress()
    max_stress = np.round(np.max(np.abs(np.array(list(stress.values())))), decimals=4)
    mass_ = t.structure_mass()
    total_mass = round(mass_[0],4)
    stw_ = np.round((max_stress / total_mass),4)

    plt.title(f"Max stress: {max_stress} | Total mass: {total_mass} | SWR: {stw_}", fontsize=14)
    plt.tick_params(direction = 'in', width = 3)
    ax.grid(True, linewidth=0.2)
    plt.xlabel("X (m)", fontsize=14)
    plt.ylabel("Y (m)", fontsize=14)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    #remove grid lines
    plt.grid(False)
    plt.show()
    plt.close()
    
# def plot_truss(t):
#     sns.set(style="whitegrid")  # Set the style of the plot using Seaborn

#     plt.figure()
#     cmap = sns.color_palette("coolwarm", as_cmap=True)  # Use a Seaborn color palette

#     nodes_dict = {node[0]: (node[1], node[2]) for node in t.nodes}

#     for node in t.nodes:
#         plt.plot(node[1], node[2], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=10)

#     for member in t.members:
#         nodes = t.members[member]
#         thickness = t._area[member] / 1.5
#         color = cmap(thickness)  # Use the colormap to determine the color based on thickness
#         plt.plot([nodes_dict[nodes[0]][0], nodes_dict[nodes[1]][0]], 
#                  [nodes_dict[nodes[0]][1], nodes_dict[nodes[1]][1]], linewidth=thickness)

#     for supp in t.supports:
#         if t.supports[supp] == "pinned":
#             plt.plot(nodes_dict[supp][0], nodes_dict[supp][1], 's', markerfacecolor='red', markeredgecolor='k', markersize=10)
#         if t.supports[supp] == "roller":
#             plt.plot(nodes_dict[supp][0], nodes_dict[supp][1], '^', markerfacecolor='green', markeredgecolor='k', markersize=10)

#     for node, load_list in t.loads.items():
#         for load in load_list:
#             magnitude, direction = load
#             x, y = nodes_dict[node]
#             if not str(magnitude).startswith("R"):
#                 angle = np.radians(float(direction))
#                 dx = (magnitude / abs(magnitude)) * np.cos(float(angle))
#                 dy = (magnitude / abs(magnitude)) * np.sin(float(angle))
#                 plt.arrow(x, y, float(dx) / 2, float(dy) / 2, head_width=0.1, color='purple')  # Use a distinct color for arrows

#     plt.show()
#     plt.close()



def make_truss(node_dict, members, load, supports):
    truss = Truss()
    for node, coord in node_dict.items():
        truss.add_node(node, *coord)
    for member, (node1, node2, iden) in members.items():
        truss.add_member(member, node1, node2, iden)
        
    for node, (x, y) in load.items():
        truss.apply_load(node, x, y)

    for node, support_type in supports.items():
        truss.apply_support(node, type=support_type)
        
    return truss


# plot_truss(t)
def convert(obj):
    """Convert unsupported JSON objects to Python native types."""
    if hasattr(obj, 'tolist'):  # Converts numpy arrays to lists
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Converts numpy numbers to Python numbers
        return obj.item()
    else:
        return str(obj)  # As a last resort, convert to string
    
def save_truss(t, node_dict, members_dict ,filename):
    stress = t.member_stress()

    for key, value in stress.items():
        stress[key] = stress[key] = value[0] if isinstance(value, np.ndarray) and value.size == 1 else value

    df = pd.DataFrame({'Member': t.member_mass.keys(), 'Stress': stress.values(), 'Mass': t.member_mass.values()})


    df = df.reset_index(drop=True)
    result_dict = df.to_dict()

    combined_dict = {"node_dict": node_dict, "member_dict": members_dict, "result": result_dict}
    print(combined_dict )
    json_str = json.dumps(combined_dict, default=convert, indent=4)

    # file_path = "./raw_results_paper/q1_p1/"+str(1)+".json"
    with open(filename, 'w') as json_file:
        json_file.write(json_str)
        json_file.close()



def parse(data_str):
    code_block = data_str.split("```python")[1].split("```")[0]

    node_dict = ast.literal_eval(code_block.split('node_dict = ')[1].split("member_dict")[0].strip())
    members_dict = ast.literal_eval(code_block.split('member_dict = ')[1].strip())
    return node_dict, members_dict

def first_three_nodes_match(dict1, dict2):
    keys_to_check = ['node_1', 'node_2', 'node_3']
    return all(tuple(dict1.get(k)) == tuple(dict2.get(k)) for k in keys_to_check)
    

def save_response(response_text: str, folder: str, identifier: str, attempts: int):
    filename = f"{folder}_{identifier}_{attempts}.txt"
    with open(filename, "w") as f:
        f.write(response_text)

def first_three_nodes_match(dict1, dict2):
    keys_to_check = ['node_1', 'node_2', 'node_3']
    return all(tuple(dict1.get(k)) == tuple(dict2.get(k)) for k in keys_to_check)