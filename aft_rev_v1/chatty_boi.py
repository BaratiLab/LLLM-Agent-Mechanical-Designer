from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, List,Tuple,Optional
from dotenv import load_dotenv
from utils import *
import os
import numpy as np
import traceback

load_dotenv()
def get_response(model="gpt-4.1-mini", message=None):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model=model,
        
        input=[

            {"role": "user", "content": message}
        ],
        instructions = "You are a concise, expert structural optimization agent specialized in 2D truss design. Generate a fully optimized, constraint-satisfying truss structure in a single shot based on user-provided nodes, supports, loads, and constraints. Use precise float values (1e-2 precision) for node coordinates.",
        temperature= 1.2,
        
    )
    return response


class TrussOutput(BaseModel):
    preamble: str
    analysis: str
    reasoning: str
    calculation: str
    node_dict: Dict[str, list[float, float]]
    member_dict: Dict[str, list[str, str, str]]
    
    # ADD OPTIONAL FIELDS FOR OPTIMALITY BOOL
    optimality: Optional[bool] = None
    scratch_pad_1: Optional[str] = None
    scratch_pad_2: Optional[str] = None
    scratch_pad_3: Optional[str] = None
    scratch_pad_final: Optional[str] = None
    stop: Optional[str] = None
    proposal: Optional[str] = None


def make_result_dict(stress, mass, print_=True, stw = True):
    total_mass = round(mass[0],4)
    member_mass = mass[1]

    member_mass = {k: round(v, 4) for k, v in mass[1].items()}
    member_stress = {k: round(v, 4) for k, v in stress.items()}

    max_stress = np.round(np.max(np.abs(np.array(list(stress.values())))), decimals=4)

    if stw == True:
        stw_ = np.round((max_stress / total_mass),4)
    
    if print_:
        print("Max stress to weight ratio: ", stw_)
        print("Max stress: ", max_stress)
        print("Total mass: ", total_mass)
        print("Member mass: ", member_mass)
        print("Member stress: ", member_stress)

    # # Create a dictionary to hold the results
    # result_dict = {
    #     "total_mass": total_mass,
    #     "max_stress": max_stress,
    #     "member_stress": stress,
    #     "member_mass": member_mass
    # }

    return f"Total mass: {total_mass}, Max stress: {max_stress}, Member_stress: {member_stress}, Member_mass: {member_mass}", {
        "max_stress_to_weight_ratio": stw_,
        "total_mass": total_mass,
        "max_stress": max_stress,
        "member_stress": member_stress,
        "member_mass": member_mass
    }

import os
def get_answer(model="gpt-4.1-mini", max_attempts = 10, message=None, broken_folder = "./broken_results/", identifier = "test", response_folder = "./responses/", load = {}, supports = {}, plot=True, print_=True, init_node_dict={}, stw_= True):
    attempts = 0

    # Check if the broken folder exists, if not create it
    if not os.path.exists(broken_folder):
        os.makedirs(broken_folder)
    # Check if the response folder exists, if not create it
    if not os.path.exists(response_folder):
        os.makedirs(response_folder)
    # Check if the response folder exists, if not create it

    while attempts < max_attempts:
        print(f"Attempt {attempts + 1} of {max_attempts}")
        try:
            response_ = get_response(model = model ,message=message)
            out_dict = ast.literal_eval(response_.output_text)
            parsed_output = TrussOutput.model_validate(out_dict)
            node_dict_g = parsed_output.node_dict
            member_dict_g = parsed_output.member_dict
            stop = parsed_output.stop
            t_gen = make_truss(node_dict_g, member_dict_g, load, supports)

            if plot is True:
                plot_truss(t_gen)
            
            res, result_dict = make_result_dict(t_gen.member_stress(), t_gen.structure_mass(), print_=print_, stw =stw_ )

            #check if the node_dict_g and init_node_dict have same first three nodal positions at load and support positions
            # raise exception if they are not same
            assert first_three_nodes_match(node_dict_g, init_node_dict), "The first three nodes do not match the initial node dictionary."

            break

        except Exception as e:
            save_response(str(response_), broken_folder, identifier, attempts)
            # traceback.print_exc()
            print(f"Error: {e}")
            #print entire error message
    
            attempts += 1
            continue

    save_response(str(response_.output_text), response_folder, identifier, attempts)
    save_response(str(response_), response_folder+"_raw", identifier, attempts)

    return response_, t_gen, attempts, node_dict_g, member_dict_g, res, result_dict, stop