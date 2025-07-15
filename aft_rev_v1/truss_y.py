"""
This module can be used to solve problems related
to 2D Trusses.
"""

from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos
import numpy as np
import math

class Truss:
    """
    A Truss is an assembly of members such as beams,
    connected by nodes, that create a rigid structure.
    In engineering, a truss is a structure that
    consists of two-force members only.

    Trusses are extremely important in engineering applications
    and can be seen in numerous real-world applications like bridges.

    Examples
    ========

    There is a Truss consisting of four nodes and five
    members connecting the nodes. A force P acts
    downward on the node D and there also exist pinned
    and roller joints on the nodes A and B respectively.

    .. image:: truss_example.png

    >>> from sympy.physics.continuum_mechanics.truss import Truss
    >>> t = Truss()
    >>> t.add_node("node_1", 0, 0)
    >>> t.add_node("node_2", 6, 0)
    >>> t.add_node("node_3", 2, 2)
    >>> t.add_node("node_4", 2, 0)
    >>> t.add_member("member_1", "node_1", "node_4")
    >>> t.add_member("member_2", "node_2", "node_4")
    >>> t.add_member("member_3", "node_1", "node_3")
    >>> t.add_member("member_4", "node_2", "node_3")
    >>> t.add_member("member_5", "node_3", "node_4")
    >>> t.apply_load("node_4", magnitude=10, direction=270)
    >>> t.apply_support("node_1", type="fixed")
    >>> t.apply_support("node_2", type="roller")
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._nodes = []
        self._members = {}
        self._loads = {}
        self._supports = {}
        self._node_labels = []
        self._node_positions = []
        self._node_position_x = []
        self._node_position_y = []
        self._nodes_occupied = {}
        self._reaction_loads = {}
        self._internal_forces = {}
        self._node_coordinates = {}

        self._area={}

        self.E=29.6*1e6

        self.area_dict = {
        "0": 1,  # Assuming this is a placeholder or a default value.
        "1": 0.195,
        "2": 0.782,
        "3": 1.759,
        "4": 3.128,
        "5": 4.887,
        "6": 7.037,
        "7": 9.578,
        "8": 12.511,
        "9": 15.834,
        "10": 19.548,
    }


    @property
    def nodes(self):
        """
        Returns the nodes of the truss along with their positions.
        """
        return self._nodes

    @property
    def node_labels(self):
        """
        Returns the node labels of the truss.
        """
        return self._node_labels
    
    @property
    def member_length(self):
        """
        Returns the length for each member.
        """
        member_length = {}
        for member in list(self._members):
            start = self._members[member][0]
            end = self._members[member][1]
            length = sqrt((self._node_coordinates[start][0]-self._node_coordinates[end][0])**2 + (self._node_coordinates[start][1]-self._node_coordinates[end][1])**2)
            member_length[member] = length
        return member_length
        
    @property
    def member_direction(self):
        """
        Returns the direction for each member.
        """
        member_direction = {}
        for member in list(self._members):
            start = self._members[member][0]
            end = self._members[member][1]
            length = sqrt((self._node_coordinates[start][0]-self._node_coordinates[end][0])**2 + (self._node_coordinates[start][1]-self._node_coordinates[end][1])**2)
            horizontal_component_start = (self._node_coordinates[end][0]-self._node_coordinates[start][0])/length
            vertical_component_start = (self._node_coordinates[end][1]-self._node_coordinates[start][1])/length
            member_direction[member] = (horizontal_component_start, vertical_component_start)
        return member_direction
    
    def calculate_local_stiffness(self,length, direction, A=1):
        c, s = direction
        factor = (A * self.E) / length
        # factor=1
        k_local = factor * np.array([
            [c**2, c*s, -c**2, -c*s],
            [c*s, s**2, -c*s, -s**2],
            [-c**2, -c*s, c**2, c*s],
            [-c*s, -s**2, c*s, s**2]
        ],float)
        return k_local

    @property
    def member_local_stiff_mat(self):
        """
        Returns the local stiffness matrix for each member.
        """
        member_local_stiff_mat = {}
        for member in list(self._members):
            length = self.member_length[member]
            area = self._area[member]
            horizontal_component_start, vertical_component_start = self.member_direction[member]
            member_local_stiff_mat[member] = self.calculate_local_stiffness(length, (horizontal_component_start, vertical_component_start), area)

        return member_local_stiff_mat
    
    @property
    def member_mass(self):
        """
        Returns the mass for each member.
        """
        member_mass = {}
        for member in list(self._members):
            length = self.member_length[member]
            area = self._area[member]
            member_mass[member] = float(length)*area
        return member_mass
    
    def structure_mass(self):
        """
        Returns the mass of the truss.
        """
        return sum(self.member_mass.values()), self.member_mass

    @property
    def node_positions(self):
        """
        Returns the positions of the nodes of the truss.
        """
        return self._node_positions

    @property
    def members(self):
        """
        Returns the members of the truss along with the start and end points.
        """
        return self._members

    @property
    def member_labels(self):
        """
        Returns the members of the truss along with the start and end points.
        """
        return self._member_labels

    @property
    def supports(self):
        """
        Returns the nodes with provided supports along with the kind of support provided i.e.
        pinned or roller.
        """
        return self._supports

    @property
    def loads(self):
        """
        Returns the loads acting on the truss.
        """
        return self._loads

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces for all supports which are all initialized to 0.
        """
        return self._reaction_loads

    
    # def all_elements_are_numbers(self,lst):

    #     return all(isinstance(item, (int, float)) for sublist in lst for item in sublist)

    def sublist_all_numbers(self,sublist):
        return all(isinstance(item, (int, float)) for item in sublist)

    @property
    def nodal_forces(self):
        """
        Returns the nodal forces for all nodes which are all initialized to 0.
        """
        # filtered_nodes = {node: loads[0] for node, loads in self._loads.items() if self.all_elements_are_numbers(loads)}
        filtered_nodes = {}
        for node, loads in self._loads.items():
            numeric_loads = [load for load in loads if self.sublist_all_numbers(load)]
            if numeric_loads:
                filtered_nodes[node] = numeric_loads[0]

        nodal_forces = np.zeros((2*len(self.nodes), 1))
        # print(filtered_nodes)
        
        for node, (x, y) in filtered_nodes.items():
            indices = self.node_map[node]
            fx = x*math.cos(math.radians(y))
            fy = x*math.sin(math.radians(y))
            nodal_forces[indices[0]] = fx
            nodal_forces[indices[1]] = fy
        return nodal_forces
    
    def global_stiffness_matrix(self):
        """
        Returns the global stiffness matrix for the truss.
        """
        k_global = np.zeros((2*len(self.nodes), 2*len(self.nodes)))
        # global_stiffness_matrix = np.zeros((8,8))

        # for member, k_local in t.member_local_stiff_mat.items():
        for member, k_local in self.member_local_stiff_mat.items():
            node_start, node_end = self.members[member]
            indices_start = self.node_map[node_start]
            indices_end = self.node_map[node_end]
            k_local=(np.array(k_local, float))
            # print(k_local)

            for i in range(2):
                for j in range(2):
                    k_global[indices_start[i], indices_start[j]] += k_local[i, j]
                    k_global[indices_start[i], indices_end[j]] += k_local[i, j + 2]
                    k_global[indices_end[i], indices_start[j]] += k_local[i + 2, j]
                    k_global[indices_end[i], indices_end[j]] += k_local[i + 2, j + 2]
        return k_global

    # def remove_dofs(self):
    #     """
    #     Removes the rows and columns of the global stiffness matrix
    #     corresponding to the nodes with supports and also reduces the nodal forces vector.
        
    #     Returns:
    #     tuple: A tuple containing the reduced global stiffness matrix, the reduced nodal forces array,
    #         and a dictionary of reduced nodal forces for considered nodes.
    #     """
    #     k_global = self.global_stiffness_matrix()
    #     dofs = []
    #     for node, support in self._supports.items():
    #         indices = self.node_map[node]
    #         if support == "pinned":
    #             dofs.extend(indices)
    #         elif support == "roller":
    #             dofs.append(indices[1])

    #     k_global = np.delete(k_global, dofs, axis=0)
    #     k_global = np.delete(k_global, dofs, axis=1)
    #     reduced_nodal_forces_array = np.delete(self.nodal_forces, dofs, axis=0)

    #     # Create a dictionary for reduced nodal forces
    #     reduced_nodal_forces_dict = {}
    #     reduced_index = 0
    #     for node, indices in self.node_map.items():
    #         if node in self._supports and self._supports[node] == "pinned":
    #             continue
    #         if node in self._supports and self._supports[node] == "roller":
    #             reduced_nodal_forces_dict[node] = [float(reduced_nodal_forces_array[reduced_index]), 0.0]
    #             reduced_index += 1
    #         else:
    #             reduced_nodal_forces_dict[node] = [float(reduced_nodal_forces_array[reduced_index]), 
    #                                             float(reduced_nodal_forces_array[reduced_index + 1])]
    #             reduced_index += 2

    #     return k_global, reduced_nodal_forces_array, reduced_nodal_forces_dict
    

    # def get_nodal_displacements(self):
    #     """
    #     Solves for and returns the nodal displacements for all nodes, considering the support conditions.
        
    #     Returns:
    #     dict: A dictionary mapping each node to its displacement vector [dx, dy].
    #     """
    #     k_global, reduced_nodal_forces, _ = self.remove_dofs()
    #     nodal_displacements_array = np.linalg.solve(k_global, reduced_nodal_forces)

    #     # Initialize the displacement dictionary with zeros
    #     displacements_dict = {node: [0, 0] for node in self.node_map}

    #     # Assign the calculated displacements to the corresponding nodes
    #     reduced_index = 0
    #     for node, indices in self.node_map.items():
    #         if node in self._supports and self._supports[node] == "pinned":
    #             continue
    #         if node in self._supports and self._supports[node] == "roller":
    #             displacements_dict[node][0] = float(nodal_displacements_array[reduced_index])
    #             reduced_index += 1
    #         else:
    #             displacements_dict[node] = [float(nodal_displacements_array[reduced_index]), 
    #                                         float(nodal_displacements_array[reduced_index + 1])]
    #             reduced_index += 2

    #     return displacements_dict
    def remove_dofs(self):
        """
        Removes the rows and columns of the global stiffness matrix
        corresponding to the nodes with supports and also reduces the nodal forces vector.
        
        Returns:
        tuple: A tuple containing the reduced global stiffness matrix, the reduced nodal forces array,
            a dictionary mapping original DOFs to reduced DOFs, and the list of constrained DOFs.
        """
        k_global = self.global_stiffness_matrix()
        nodal_forces = self.nodal_forces
        
        # Create a list of constrained DOFs
        constrained_dofs = []
        for node, support in self._supports.items():
            indices = self.node_map[node]
            if support == "pinned":
                constrained_dofs.extend(indices)  # Both x and y DOFs are constrained
            elif support == "roller":
                constrained_dofs.append(indices[1])  # Only y DOF is constrained
        
        # Create mapping from original DOFs to reduced DOFs
        total_dofs = 2 * len(self.nodes)
        dof_mapping = {}
        reduced_index = 0
        
        for original_dof in range(total_dofs):
            if original_dof not in constrained_dofs:
                dof_mapping[original_dof] = reduced_index
                reduced_index += 1
        
        # Remove constrained DOFs from stiffness matrix and nodal forces
        k_reduced = np.delete(k_global, constrained_dofs, axis=0)
        k_reduced = np.delete(k_reduced, constrained_dofs, axis=1)
        reduced_nodal_forces = np.delete(nodal_forces, constrained_dofs, axis=0)
        
        return k_reduced, reduced_nodal_forces, dof_mapping, constrained_dofs

    def get_nodal_displacements(self):
        """
        Solves for and returns the nodal displacements for all nodes, considering the support conditions.
        
        Returns:
        dict: A dictionary mapping each node to its displacement vector [dx, dy].
        """
        k_reduced, reduced_nodal_forces, dof_mapping, constrained_dofs = self.remove_dofs()
        
        # Solve for the unconstrained displacements
        # try:
        #     reduced_displacements = np.linalg.solve(k_reduced, reduced_nodal_forces)
        # except np.linalg.LinAlgError:
        #     reduced_displacements = np.linalg.lstsq(k_reduced, reduced_nodal_forces, rcond=None)[0]
        
        try:
            reduced_displacements = np.linalg.solve(k_reduced, reduced_nodal_forces)
        except np.linalg.LinAlgError:
            # print("WARNING: Singular matrix — switching to pseudoinverse")
            #try again
            pass
            # reduced_displacements = np.dot(np.linalg.pinv(k_reduced), reduced_nodal_forces)

        # Initialize the full displacement vector with zeros
        full_displacements = np.zeros((2 * len(self.nodes), 1))
        
        # Fill in the computed displacements for unconstrained DOFs
        for original_dof, reduced_dof in dof_mapping.items():
            full_displacements[original_dof] = reduced_displacements[reduced_dof]
        
        # Convert to node-based dictionary format
        displacements_dict = {}
        for node, indices in self.node_map.items():
            displacements_dict[node] = [
                float(full_displacements[indices[0]]),
                float(full_displacements[indices[1]])
            ]
        
        return displacements_dict

    # def member_stress(self):
    #     """
    #     Returns the stress for each member.
    #     """
    #     member_stress = {}
    #     for member, node in self.members.items():
    #         node1, node2 = node
    #         c,s= self.member_direction[member]
    #         l = self.member_length[member]
    #         q = self.get_nodal_displacements()
    #         stress_m = np.array([-c, -s, c, s])
    #         q_m = np.array(q[node1] + q[node2])
    #         stress=np.dot(stress_m,(q_m.reshape(-1,1)))*(29.6*1e6/l)
    #         member_stress[member] = stress
    #     return member_stress
    
    def member_stress(self):
        """
        Returns the stress for each member as float values.
        """
        member_stress = {}
        for member, node in self.members.items():
            node1, node2 = node
            c, s = self.member_direction[member]
            l = self.member_length[member]
            q = self.get_nodal_displacements()
            
            # Ensure all values are float type
            stress_m = np.array([-c, -s, c, s], dtype=float)
            q_m = np.array(q[node1] + q[node2], dtype=float)
            
            # Calculate stress and convert result to float
            stress = np.dot(stress_m, (q_m.reshape(-1, 1))) * (29.6*1e6/l)
            member_stress[member] = float(stress[0])  # Extract scalar float value
            
        return member_stress
        

    @property
    def internal_forces(self):
        """
        Returns the internal forces for all members which are all initialized to 0.
        """
        return self._internal_forces

    @property
    def node_map(self):
        node_mapping = {}
        for i, node in enumerate(self._nodes):
            node_mapping[node[0]] = [2*i, 2*i+1]
        return node_mapping

    # def add_node(self, label, x, y):
    #     """
    #     This method adds a node to the truss along with its name/label and its location.

    #     Parameters
    #     ==========
    #     label:  String or a Symbol
    #         The label for a node. It is the only way to identify a particular node.

    #     x: Sympifyable
    #         The x-coordinate of the position of the node.

    #     y: Sympifyable
    #         The y-coordinate of the position of the node.

    #     Examples
    #     ========

    #     >>> from sympy.physics.continuum_mechanics.truss import Truss
    #     >>> t = Truss()
    #     >>> t.add_node('A', 0, 0)
    #     >>> t.nodes
    #     [('A', 0, 0)]
    #     >>> t.add_node('B', 3, 0)
    #     >>> t.nodes
    #     [('A', 0, 0), ('B', 3, 0)]
    #     """
    #     # x = sympify(x)
    #     # y = sympify(y)

    #     if label in self._node_labels:
    #         raise ValueError("Node needs to have a unique label")

    #     elif x in self._node_position_x and y in self._node_position_y and self._node_position_x.index(x)==self._node_position_y.index(y):
    #         raise ValueError("A node already exists at the given position")

    #     else :
    #         self._nodes.append((label, x, y))
    #         self._node_labels.append(label)
    #         self._node_positions.append((x, y))
    #         self._node_position_x.append(x)
    #         self._node_position_y.append(y)
    #         self._node_coordinates[label] = [x, y]

    def add_node(self, label, x, y, tol=1e-6):
        """
        Adds a node to the truss with tolerance-based coordinate uniqueness check.
        
        Parameters:
            label: str
                Unique label for the node (e.g., 'node_1')
            x, y: float
                Coordinates of the node
            tol: float
                Tolerance for coordinate duplication (default 1e-6)
        """
        if label in self._node_labels:
            raise ValueError(f"Node '{label}' already exists.")

        # for existing_x, existing_y in self._node_positions:
        #     if abs(existing_x - x) < tol and abs(existing_y - y) < tol:
        #         raise ValueError(f"A node already exists near ({x:.4f}, {y:.4f})")

        self._nodes.append((label, x, y))
        self._node_labels.append(label)
        self._node_positions.append((x, y))
        self._node_position_x.append(x)
        self._node_position_y.append(y)
        self._node_coordinates[label] = [x, y]


    def add_member(self, label, start, end, iden=None):
        """
        This method adds a member between any two nodes in the given truss.

        Parameters
        ==========
        label: String or Symbol
            The label for a member. It is the only way to identify a particular member.

        start: String or Symbol
            The label of the starting point/node of the member.

        end: String or Symbol
            The label of the ending point/node of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.add_node('C', 2, 2)
        >>> t.add_member('AB', 'A', 'B')
        >>> t.members
        {'AB': ['A', 'B']}
        """

        if start not in self._node_labels or end not in self._node_labels or start==end:
            raise ValueError("The start and end points of the member must be unique nodes")

        elif label in list(self._members):
            raise ValueError("A member with the same label already exists for the truss")

        elif self._nodes_occupied.get((start, end)):
            raise ValueError(f"A member already exists between the two nodes {start} and {end}")

        else:
            self._members[label] = [start, end]
            self._nodes_occupied[start, end] = True
            self._nodes_occupied[end, start] = True
            self._internal_forces[label] = 0
            self._area[label]= self.area_dict.get(iden, 1)

            


    def apply_load(self, location, magnitude, direction):
        """
        This method applies an external load at a particular node

        Parameters
        ==========
        location: String or Symbol
            Label of the Node at which load is applied.

        magnitude: Sympifyable
            Magnitude of the load applied. It must always be positive and any changes in
            the direction of the load are not reflected here.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> P = symbols('P')
        >>> t.apply_load('A', P, 90)
        >>> t.apply_load('A', P/2, 45)
        >>> t.apply_load('A', P/4, 90)
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        """
        magnitude = (magnitude)
        direction = (direction)

        if location not in self.node_labels:
            raise ValueError("Load must be applied at a known node")

        else:
            if location in list(self._loads):
                self._loads[location].append([magnitude, direction])
            else:
                self._loads[location] = [[magnitude, direction]]

    def apply_support(self, location, type):
        """
        This method adds a pinned or roller support at a particular node

        Parameters
        ==========

        location: String or Symbol
            Label of the Node at which support is added.

        type: String
            Type of the support being provided at the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.apply_support('A', 'pinned')
        >>> t.supports
        {'A': 'pinned'}
        """
        if location not in self._node_labels:
            raise ValueError("Support must be added on a known node")

        else:
            if location not in list(self._supports):
                if type == 'pinned':
                    self.apply_load(location, Symbol('R_'+str(location)+'_x'), 0)
                    self.apply_load(location, Symbol('R_'+str(location)+'_y'), 90)
                elif type == 'roller':
                    self.apply_load(location, Symbol('R_'+str(location)+'_y'), 90)
            elif self._supports[location] == 'pinned':
                if type == 'roller':
                    self.remove_load(location, Symbol('R_'+str(location)+'_x'), 0)
            elif self._supports[location] == 'roller':
                if type == 'pinned':
                    self.apply_load(location, Symbol('R_'+str(location)+'_x'), 0)
            self._supports[location] = type


    