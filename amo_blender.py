from bpy_extras.io_utils import (ImportHelper,
                                 ExportHelper,
                                 unpack_list,
                                 unpack_face_list,
                                 axis_conversion,
                                 )
from bpy.props import (BoolProperty,
                       FloatProperty,
                       StringProperty,
                       EnumProperty,
                       )
import os
import math
import numpy as np
import bpy
bl_info = {
    "name": "AmoAddon",
    "description": "Import and export models as dot-amo",
    "author": "VasallSoftware",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "File > Import-Export",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/Scripts/My_Script",
    "tracker_url": "https://developer.blender.org/maniphest/task/edit/form/2/",
    "support": "COMMUNITY",
    "category": "Import-Export"
}


class ExportAMO(bpy.types.Operator, ExportHelper):
    """Save objects in AMO-format"""
    bl_idname = "export_mesh.amo"
    bl_label = "Export Dot-AMO"
    filter_glob = StringProperty(
        default="*.amo",
        options={'HIDDEN'},
    )
    check_extension = True
    filename_ext = ".amo"

    def execute(self, context):
        return save(self, context, self.filepath)


def menu_func_export(self, context):
    self.layout.operator(ExportAMO.bl_idname, text="Dot-AMO (.amo)")


classes = (
    ExportAMO,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


def vtx_nrm(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


class Vertex:
    def __init__(self, index, position):
        self.index = index
        self.position = position
        self.joints = []
        self.weights = []
        
    def add_joint(self, joint, weight):
        self.joints.append(joint)
        self.weights.append(weight)

class Joint:
    def __init__(self, index, name, parent, vertices):
        self.index = index
        self.name = name
        self.parent = parent
        self.vertices = vertices

class AMOModel:
    def __init__(self):
        self.mobj = {}
        self.marm = {}

        self.obj_verts = []
        self.obj_group_names = []

        self.vtx_arr = []
        self.vtx_num = 0
        self.uv_arr = []
        self.nrm_arr = []
        self.idx_arr = []
        self.idx_num = 0
        
        self.bone_arr = []

    def load_mesh(self):
        mesh = self.mobj.data
        
        # Collect and store the indices
        for f in mesh.polygons:
            self.idx_arr.append([f.index, 0, f.vertices[0]])
            self.idx_arr.append([f.index, 1, f.vertices[1]])
            self.idx_arr.append([f.index, 2, f.vertices[2]])

        # Collect and store all vertices
        for i in range(len(mesh.vertices)):
            # Convert position to world space
            v = self.mobj.matrix_world @ mesh.vertices[i].co
            self.vtx_arr.append(Vertex(len(self.vtx_arr), v))

        # Collect and store the UV coordinates
        same = -1
        for i in range(len(self.vtx_arr)):
            tmp = i
            same = -1

            uv = mesh.uv_layers.active.data[self.idx_arr[tmp][0] * 3 + self.idx_arr[tmp][1]].uv

            for j in range(len(self.uv_arr)):
                if(self.uv_arr[j] == uv):
                    same = j
                    break

            if(same < 0):
                self.uv_arr.append(uv)
                self.idx_arr[tmp].append(len(self.uv_arr))

            else:
                self.idx_arr[tmp].append(same + 1)

    def calc_normals(self):
        # Calculate normal vectors
        same = -1
        for v in range(0, len(self.idx_arr), 3):
            tmp = v
            arr = [self.idx_arr[tmp][2], self.idx_arr[tmp + 1][2], self.idx_arr[tmp + 2][2]]

            a = self.vtx_arr[arr[0]].position - self.vtx_arr[arr[1]].position
            b = self.vtx_arr[arr[2]].position - self.vtx_arr[arr[1]].position

            # TODO add world matrix
            nrm = np.cross(a, b)
            nrm = vtx_nrm(nrm)

            same = -1
            for i in range(len(self.nrm_arr)):
                if((nrm == self.nrm_arr[i]).all()):
                    same = i
                    break

            if same < 0:
                self.nrm_arr.append(nrm)

                idx = len(self.nrm_arr)
                self.idx_arr[tmp].append(idx)
                self.idx_arr[tmp + 1].append(idx)
                self.idx_arr[tmp + 2].append(idx)

            else:
                self.idx_arr[tmp].append(same + 1)
                self.idx_arr[tmp + 1].append(same + 1)
                self.idx_arr[tmp + 2].append(same + 1)
                
    def load_bones(self):
        for b in self.marm.pose.bones:
            # Set the parent node
            parent = b.parent
            parent_idx = -1
            if parent is not None:
                for p in self.bone_arr:
                    if p.name == parent:
                        parent_idx = p.index
                        break

            self.bone_arr.append(Joint(len(self.bone_arr), b.name, parent_idx, []))
            
    def get_bone(self, name):
        for b in self.bone_arr:
            if b.name == name:
                return b.index
            
        return -1
            
    def link_bones(self):
            for bone in self.marm.pose.bones:
                if bone.name not in self.obj_group_names:
                    continue
                
                bidx = self.get_bone(bone.name)

                gidx = self.mobj.vertex_groups[bone.name].index

                bone_verts = [v for v in self.obj_verts if gidx in [g.group for g in v.groups]]

                for v in bone_verts:
                    for g in v.groups:
                        if g.group == gidx:
                            self.vtx_arr[v.index].add_joint(bidx, g.weight)
                
    def load(self, obj, arm):
        self.mobj = obj
        self.marm = arm

        print(self.mobj)
        print(self.marm)

        self.obj_verts = self.mobj.data.vertices
        self.obj_group_names = [g.name for g in self.mobj.vertex_groups]

        # Change to object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        
        # Load the mesh
        self.load_mesh()
        
        # Calculate the normal-vectors
        self.calc_normals()

        # Change to object mode
        bpy.ops.object.mode_set(mode="POSE")

        # Load the bones
        self.load_bones()
        
        # Link the bones to the vertices
        self.link_bones()


def save(operator, context, filepath):
    # Get all selected objects
    objects = context.selected_objects

    # Get the model
    mdl = None
    for i in objects:
        if i.type == "MESH":
            mdl = i
            break

    # Get the armature
    arm = None
    for i in objects:
        print(i.type)
        if i.type == "ARMATURE":
            arm = i
            break
        
    amo = AMOModel()
    amo.load(mdl, arm)
    
    
    print("Loaded", len(amo.bone_arr), "bones!")
    for b in amo.bone_arr:
        print(b.idx, ":", len(b.vertices))        
    
    return {'FINISHED'}

    filepath = os.fsencode(filepath)
    of = open(filepath, "w")
    of.write("o test\n")

    # Write vertices
    for v in vtx_arr:
        of.write("v %.4f %.4f %.4f\n" % (v.x, v.z, v.y))

    # Write uv-coords
    for uv in uv_arr:
        of.write("vt %.4f %.4f\n" % (uv.x, uv.y))

    # Write normals
    for n in nrm_arr:
        of.write("vn %.4f %.4f %.4f\n" % (n[0], n[2], n[1]))

    # Write indices
    for i in range(0, len(idx_arr), 3):
        of.write("f ")
        of.write("%d/%d/%d " %
                 (idx_arr[i][2] + 1, idx_arr[i][3], idx_arr[i][4]))
        of.write("%d/%d/%d " %
                 (idx_arr[i + 1][2] + 1, idx_arr[i + 1][3], idx_arr[i + 1][4]))
        of.write("%d/%d/%d " %
                 (idx_arr[i + 2][2] + 1, idx_arr[i + 2][3], idx_arr[i + 2][4]))
        of.write("\n")

    of.close()

    return {'FINISHED'}


if __name__ == "__main__":
    # Get all selected objects
    objects = bpy.context.selected_objects

    # Get the model
    mdl = None
    for i in range(len(objects)):
        if objects[i].type == "MESH":
            mdl = objects[i]
            break

    # Get the armature
    arm = None
    for i in range(len(objects)):
        if objects[i].type == "ARMATURE":
            arm = objects[i]
            break
        
    amo = AMOModel()
    amo.load(mdl, arm)
    
    for v in amo.vtx_arr:
        print("Vertex %d: " % (v.index), end='')
        
        for j in range(len(v.joints)):
            print("%d:%.4f  " % (v.joints[j], v.weights[j]), end='')
            
        print("")