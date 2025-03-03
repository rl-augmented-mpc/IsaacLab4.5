import numpy as np
from pxr import Gf
from .utils.prim_util import setTranslate, setRotateXYZ, createXform, createSphere, applyRigidBody, applyMass, createFixedJoint, applyCollider

def create_rigid_point(stage, prim_path:str, position:tuple[float,float,float], radius:float):
    path, prim = createXform(stage, prim_path)
    # Creates a sphere
    sphere_path = path + "/" + "point"
    sphere_path, sphere_geom = createSphere(
        stage, sphere_path, radius, 1
    )
    setTranslate(prim, Gf.Vec3d(position))
    setRotateXYZ(prim, Gf.Vec3d(0, 0, 0))
    return path, prim

def create_foot_contact_links(stage, prim_path:str, num_point_x:int, num_point_y:int, foot:str):
    if foot == "L":
        base_position = (-0.0135, 0.098, -0.55)
    elif foot == "R":
        base_position = (-0.0135, -0.098, -0.55)
    
    foot_x = 0.15
    foot_y = 0.1
    edge_x = foot_x/2 - (foot_x/2)/num_point_x
    edge_y = foot_y/2 - (foot_y/2)/num_point_y
    if num_point_x > 1:
        x_positions = np.linspace(base_position[0]-edge_x, base_position[0]+edge_x, num_point_x)
    else:
        x_positions = np.array([base_position[0]])
    if num_point_y > 1:
        y_positions = np.linspace(base_position[1]-edge_y, base_position[1]+edge_y, num_point_y)
    else:
        y_positions = np.array([base_position[1]])
    XX, YY = np.meshgrid(x_positions, y_positions)
    contact_positions = np.vstack([XX.ravel(), YY.ravel()]).T
    contact_positions = np.hstack([contact_positions, base_position[2]*np.ones((contact_positions.shape[0], 1))]) #add z
    for n in range(len(contact_positions)):
        path, prim = create_rigid_point(stage, prim_path+f"/{foot}_contact_point_{n}", contact_positions[n].tolist(), 0.001)
        applyRigidBody(prim)
        applyMass(prim, 0.001)
        createFixedJoint(stage, prim_path+f"/{foot}_toe/{foot}_contact_point_joint_{n}", prim_path+f"/{foot}_toe", prim_path+f"/{foot}_contact_point_{n}")
        
def create_foot_hsr_contact_links(stage, prim_path:str, num_point_x:int, num_point_y:int, foot:str):
    if foot == "L":
        base_position = (0.08, 0.098, -0.55)
    elif foot == "R":
        base_position = (0.08, -0.098, -0.55)
    if num_point_x > 1:
        x_positions = np.linspace(base_position[0]-0.06, base_position[0]+0.06, num_point_x)
    else:
        x_positions = np.array([base_position[0]])
    if num_point_y > 1:
        y_positions = np.linspace(base_position[1]-0.05, base_position[1]+0.05, num_point_y)
    else:
        y_positions = np.array([base_position[1]])
    XX, YY = np.meshgrid(x_positions, y_positions)
    contact_positions = np.vstack([XX.ravel(), YY.ravel()]).T
    contact_positions = np.hstack([contact_positions, base_position[2]*np.ones((contact_positions.shape[0], 1))]) #add z
    for n in range(len(contact_positions)):
        path, prim = create_rigid_point(stage, prim_path+f"/{foot}_contact_point_{n}", contact_positions[n].tolist(), 0.001)
        applyRigidBody(prim)
        applyMass(prim, 0.001)
        createFixedJoint(stage, prim_path+f"/{foot}_toe/{foot}_contact_point_joint_{n}", prim_path+f"/{foot}_toe", prim_path+f"/{foot}_contact_point_{n}")