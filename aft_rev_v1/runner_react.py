import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from truss_y import *
from utils import *
from loader_study import *
from openai_runner import *
from tqdm import tqdm
import ast
import glob2
import traceback
import os
import json

def simple_react(node_dict, load, supports, example_members, area_id, num_iterations=10, max_mass=30, max_stress_all=10, file_name="./raw_results_paper/q1_p1_run1", content=None):

    og_node_dict = node_dict.copy()

    if not os.path.exists(file_name+"/"):
        os.makedirs(file_name+"/")

    # if content is None:
    #     content=[
                
    #             {
    #             "type": "text",
    #             "text": f"Think step by step like an engineer where to add nodes and how to connect members to Generate an optimized closed truss structure starting with {node_dict}. Add members and nodes, given loads = {load} and supports = {supports}. Develop a member_dict similar to {example_members}, utilizing cross-sections from {area_id}. Ensure correct connections and unique members. Optimize the structure to maintain maximum compressive or tensile stress below {max_stress_all} (positive for tensile and negative for compressive) and total mass under {max_mass}, calculated by summing the product of member lengths and values of {area_id}. Provide ONLY node_dict and member_dict without comments in a python code block. Remember these specifications and requirements. DO NOT change the given node_dict, loads and support positions, you can add more to it but can't change it."
    #             },
                
    #         ]
    
    if content is None:
        content=[
                
                {
                "type": "text",
                "text": f"Generate an optimized truss structure by starting with an existing set of nodes defined in node_dict. The task involves: \n Analyzing the given set of nodes positions {node_dict} and the loads {load} and supports {supports}. \n Strategically add new nodes and members to enhance the structure's strength and efficiency. Ensure any additions adhere to the given constraints without altering the initial nodes. \n Develop a member_dict similar to {example_members}, utilizing cross-sections from {area_id}. \n Design the truss to keep the maximum compressive or tensile stress below {max_stress_all} (positive for tensile and negative for compressive) and the total mass under {max_mass}. Calculate mass by summing the products of member lengths and cross-sectional areas from area_id. \n Provide ONLY node_dict and member_dict without comments in a python code block. Remember to adhere to the specifications and requirements."
                },
                
            ]


    for i in range(num_iterations):
        print(f"\nIteration {i}")
        sucess = False
        data_str, raw_res = get_ans(content, debug=True)

        with open (f"{file_name}/{i}_raw_response.txt", "w") as f:
                f.write(raw_res.text)
                f.close()

        try:
            generated_node_dict, members_dict = parse(data_str)
        except:
            error = "Error in parsing, generated content is not in correct format"
            print(error)
        
        try:
            t= make_truss(generated_node_dict, members_dict, load, supports)
            plot_truss(t)

            stress = (t.member_stress())
            max_stress = max(stress, key=lambda k: abs(stress[k]))
            print(f"Max stress is {(stress[max_stress])} in member {max_stress}, mass is {t.strucutre_mass()}")

            save_truss(t, generated_node_dict, members_dict, f"{file_name}/{i}.json")

            if abs(stress[max_stress]) <=max_stress_all and t.strucutre_mass() <= max_mass:
                print("Specifications acheived!")
                sucess = True
                break
            else:
                content.append(
                    {
                    "type": "text",
                    "text": f"You have not achieved your goal. \n Generated structure with {generated_node_dict} and {members_dict} has mass of {t.strucutre_mass()} with maximum stress value being {stress[max_stress]} (positive for tensile and negative for compressive) in member {max_stress}. \nThe stress in each memeber is {stress}.\n The weight of each member is {t.member_mass}. \n Think step by step where to add new nodes and how to connect members strategically to given {og_node_dict}, load {load} and supports {supports}, utilizing cross-sections from {area_id}, to create a structure with maximum absolute stress (tensile and compressive) under {max_stress_all} (positive for tensile and negative for compressive) and total mass under {max_mass}, put your resoning in as comments.\n Remember Thicker cross section has less stress concentration but more mass. DO NOT modify the original given node positions {og_node_dict}, you can add more nodes to it. Each addition or change should be reasoned in comments in the code. Provide ONLY node_dict and member_dict. You can choose to optimize one of the previous structures or start from scratch."
                    },
                    
                )

        except:
            error = traceback.format_exc()
            print(error)
            

    return [i, sucess]



def simple_react_stw(node_dict, load, supports, example_members, area_id, num_iterations=10, stw=1, max_mass=30, file_name="./raw_results_paper/q1_p1_run1", content=None):

    if not os.path.exists(file_name+"/"):
        os.makedirs(file_name+"/")


    og_node_dict = node_dict.copy()

    if content is None:
        # content=[
                
            #     {
            #     "type": "text",
            #     "text": f"Think step by step like an engineer where to add nodes and how to connect members to Generate an optimized closed truss structure starting with {node_dict}. Add members and nodes, aligning with {load} and {supports} for loads and supports. Develop a member_dict similar to {example_members}, utilizing cross-sections from {area_id}. Ensure correct connections and unique members. Optimize the structure to achieve stress to weight ratio of {stw} or lower and total mass under {max_mass}, calculated by summing the product of member lengths and values of {area_id}. Provide ONLY node_dict and member_dict without comments in a python code block."
            #     },
                
            # ]
        content=[
                
                {
                "type": "text",
                "text": f"Generate an optimized truss structure by starting with an existing set of nodes defined in node_dict. The task involves: \n Analyzing the given set of nodes positions {node_dict} and the loads {load} and supports {supports}. \n Strategically add new nodes and members to enhance the structure's strength and efficiency. Ensure any additions adhere to the given constraints without altering the initial nodes. \n Develop a member_dict similar to {example_members}, utilizing cross-sections from {area_id}. \n Design the truss to achieve stress-to-weight ratio of {stw} or lower and the total mass under {max_mass}. Calculate mass by summing the products of member lengths and cross-sectional areas from area_id. \n Provide ONLY node_dict and member_dict without comments in a python code block. Remember to adhere to the specifications and requirements."
                },
                
            ]
    
    for i in range(num_iterations):
        print(f"\nIteration {i}")
        sucess = False
        data_str, raw_res = get_ans(content, debug=True)

        with open (f"{file_name}/{i}_raw_response.txt", "w") as f:
                f.write(raw_res.text)
                f.close()

        try:
            generated_node_dict, members_dict = parse(data_str)
        except:
            error = "Error in parsing, generated content is not in correct format"
        
        try:
            t= make_truss(generated_node_dict, members_dict, load, supports)
            plot_truss(t)

            stress = (t.member_stress())
            max_stress = max(stress, key=lambda k: abs(stress[k]))

            score = abs(stress[max_stress])/t.strucutre_mass()

            print(f"Score {score}\nMax stress is {abs(stress[max_stress])} in member {max_stress}, mass is {t.strucutre_mass()}")

            save_truss(t, generated_node_dict, members_dict, f"{file_name}/{i}.json")

            if score <= stw and t.strucutre_mass() <= max_mass:
                print("Specifications acheived!")
                sucess = True
                break

            elif t.strucutre_mass() > max_mass:
                content.append(
                    {
                    "type": "text",
                    "text": f"You have not achieved your goal.\n Generated structure with {generated_node_dict} and {members_dict} has mass of {t.strucutre_mass()} which is greater than the threshold value of {max_mass}. \n Remember Thicker cross section has less stress concentration but more mass.\n Think step by step where to add nodes and how to connect members and create a structure and ensure the structure has stress to weight ratio of {stw} and total mass under {max_mass}. "
                    },
                    
                )


            elif score > stw: 
                # content.append(
                #     {
                #     "type": "text",
                #     "text": f"You have not achieved your goal. Generated structure with {node_dict} and {members_dict} has stress to weight ration of {score} which is greater than the threhold value of {stw}. Think step by step where to add nodes and how to connect members and create a structure and ensure the structure has stress to weight ratio of {stw} and total mass under {max_mass}."
                #     },
                    
                # )

                content.append(
                    {
                    "type": "text",
                    "text": f"You have not achieved your goal. \n Generated structure with {generated_node_dict} and {members_dict} has stress-to-weight ratio of {score}. \nThe stress in each memeber is {stress}.\n The weight of each member is {t.member_mass}. \n Think step by step where to add new nodes and how to connect members strategically to given {og_node_dict}, load {load} and supports {supports}, utilizing cross-sections from {area_id}, to create a structure with stress-to-weight ratio of {stw} or lower  and total mass under {max_mass}, put your resoning in as comments.\n Remember Thicker cross section has less stress concentration but more mass. DO NOT modify the original given node positions {og_node_dict}, you can add more nodes to it. Each addition or change should be reasoned in comments in the code. Provide ONLY node_dict and member_dict. You can choose to optimize one of the previous structures or start from scratch, You can choose to increase the cross sectional area in members with large stress concentration."
                    },
                )
                
                if len(content) > 5:
                    content.pop(1)

        except:
            error = traceback.format_exc()
            print(error)
            continue

    return [i, sucess]

