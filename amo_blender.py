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

import os
import math
import numpy as np
import bpy
import math

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

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


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
    def __init__(self, index, name, parent):
        self.index = index
        self.name = name
        self.parent = parent
        self.matrix = []

class BoneKeyframe:
    def __init__(self, index):
        self.index = index
        
        self.loc = []
        self.rot = []
        
    def add_loc(self, val):
        self.loc.append(val)
        
    def add_rot(self, val):
        self.rot.append(val)

class Keyframe:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.bones = []
        
    def add_key(self, bone, flag, value):
        found = -1
        for b in range(len(self.bones)):
            if self.bones[b].index == bone:
                found = b
                break
            
        if found >= 0:
            if flag == 0:
                self.bones[found].add_loc(value)
            elif flag == 1:
                self.bones[found].add_rot(value)
                
        else:
            key = BoneKeyframe(bone)
            
            if flag == 0:
                key.add_loc(value)
            elif flag == 1:
                key.add_rot(value)
                
            self.bones.append(key)
               
class Animation:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        
        self.keyframes = []
        
    def add_value(self, timestamp, bone, flag, value):
        found = -1
        
        for k in range(len(self.keyframes)):
            if self.keyframes[k].timestamp == timestamp:
                found = k
                break
            
        if found >= 0:
            self.keyframes[found].add_key(bone, flag, value)
            
        else:
            key = Keyframe(timestamp)
            
            key.add_key(bone, flag, value)
            
            self.keyframes.append(key)
            

class AMOModel:
    def __init__(self):
        self.mobj = {}
        self.marm = {}

        self.obj_verts = []
        self.obj_group_names = []

        self.vtx_arr = []
        self.tex_arr = []
        self.nrm_arr = []
        self.jnt_arr = []
        self.wgt_arr = []
        self.idx_arr = []
        
        self.bone_arr = []
        self.anim_arr = []

    def load_mesh(self):
        mesh = self.mobj.data
        
        if len(mesh.polygons[0].vertices) == 3:
            for f in mesh.polygons:
                self.idx_arr.append([f.index, 0, f.vertices[0]])
                self.idx_arr.append([f.index, 1, f.vertices[1]])
                self.idx_arr.append([f.index, 2, f.vertices[2]])
        else:
            for f in mesh.polygons:
                self.idx_arr.append([f.index, 0, f.vertices[3]])
                self.idx_arr.append([f.index, 1, f.vertices[0]])
                self.idx_arr.append([f.index, 2, f.vertices[1]])
            
            for f in mesh.polygons:
                self.idx_arr.append([f.index, 0, f.vertices[3]])
                self.idx_arr.append([f.index, 1, f.vertices[1]])
                self.idx_arr.append([f.index, 2, f.vertices[2]])

        # Collect and store all vertices
        for i in range(len(mesh.vertices)):
            # Convert position to world space
            v = self.mobj.matrix_world @ mesh.vertices[i].co
            self.vtx_arr.append(Vertex(len(self.vtx_arr), v))

        # Collect and store the UV coordinates
        same = -1
        for i in range(len(self.idx_arr)):
            same = -1

            tex = mesh.uv_layers.active.data[self.idx_arr[i][0] * 3 + self.idx_arr[i][1]].uv

            for j in range(len(self.tex_arr)):
                if(self.tex_arr[j] == tex):
                    same = j
                    break

            if(same < 0):
                idx = len(self.tex_arr)
                self.tex_arr.append(tex)
                self.idx_arr[i].append(idx)

            else:
                self.idx_arr[i].append(same)

    def calc_normals(self):
        # Calculate normal vectors
        same = -1
        for v in range(0, len(self.idx_arr), 3):
            arr = [self.idx_arr[v][2], self.idx_arr[v + 1][2], self.idx_arr[v + 2][2]]

            a = self.vtx_arr[arr[1]].position - self.vtx_arr[arr[0]].position
            b = self.vtx_arr[arr[1]].position - self.vtx_arr[arr[2]].position

            # TODO add world matrix
            nrm = np.cross(a, b)
            nrm = vtx_nrm(nrm)
            nrm = nrm * -1.0

            same = -1
            for i in range(len(self.nrm_arr)):
                if((nrm == self.nrm_arr[i]).all()):
                    same = i
                    break

            if same < 0:
                idx = len(self.nrm_arr)
                self.nrm_arr.append(nrm)
                
                self.idx_arr[v].append(idx)
                self.idx_arr[v + 1].append(idx)
                self.idx_arr[v + 2].append(idx)

            else:
                self.idx_arr[v].append(same)
                self.idx_arr[v + 1].append(same)
                self.idx_arr[v + 2].append(same)           
    
    def load_bones(self):
        for b in self.marm.pose.bones:
            # Set the parent node
            parent = b.parent
            parent_idx = -1
            if parent is not None:
                for p in self.bone_arr:
                    if p.name == parent.name:
                        parent_idx = p.index
                        break
            
            self.bone_arr.append(Joint(len(self.bone_arr), b.name, parent_idx))
            
    def get_bone(self, name):
        for b in self.bone_arr:
            if b.name == name:
                return b.index
            
        return -1
            
    def link_bones(self):
            for bone in self.marm.pose.bones:                
                bidx = self.get_bone(bone.name)

                gidx = self.mobj.vertex_groups[bone.name].index

                bone_verts = [v for v in self.obj_verts if gidx in [g.group for g in v.groups]]

                for v in bone_verts:
                    for g in v.groups:
                        if g.group == gidx:
                            self.vtx_arr[v.index].add_joint(bidx, g.weight)
                            
        
        
    def calc_mat(self, name):
        local = self.marm.data.bones[name].matrix_local
        basis = self.marm.pose.bones[name].matrix_basis
            
        parent = self.marm.pose.bones[name].parent
        if parent == None:
            return local @ basis
        else:
            parent_local = self.marm.data.bones[parent.name].matrix_local
            return self.calc_mat(parent.name) @ (parent_local.inverted() @ local) @ basis
                            
    def calc_joint_matrices(self):
        for b in self.bone_arr:
            b.matrix = self.calc_mat(b.name)
                            
    def fill_joints(self):
        for i in range(len(self.idx_arr)):
            jnt_arr = self.vtx_arr[self.idx_arr[i][2]].joints
            wgt_arr = self.vtx_arr[self.idx_arr[i][2]].weights
            
            if len(jnt_arr) < 4:
                for j in range(0, 4 - len(jnt_arr), 1):
                    jnt_arr.append(-1)
                    wgt_arr.append(0.0)
                    
            elif len(jnt_arr) > 4:
                jnt_arr = jnt_arr[:4]
                wgt_arr = wgt_arr[:4]
            
            same = -1
            for j in range(len(self.jnt_arr)):
                if self.jnt_arr[j] == jnt_arr:
                    same = j
                    break
                
            if same >= 0:
                self.idx_arr[i].append(same)
                
            else:
                idx = len(self.jnt_arr)
                self.jnt_arr.append(jnt_arr)
                self.idx_arr[i].append(idx)
                
            same = -1
            for w in range(len(self.wgt_arr)):
                if self.wgt_arr[w] == wgt_arr:
                    same = w
                    break
                
            if same >= 0:
                self.idx_arr[i].append(same)
                
            else:
                idx = len(self.wgt_arr)
                self.wgt_arr.append(wgt_arr)
                self.idx_arr[i].append(idx)
                            
    def load_animation(self):
        for action in bpy.data.actions:
            anim = Animation(len(self.anim_arr), action.name)
            
            for fc in action.fcurves:
                bone = fc.data_path.split('"')[1]
                bone_idx = self.get_bone(bone)
                flag = -1
                
                if fc.data_path.endswith("location"):
                    flag = 0
                    
                elif fc.data_path.endswith("rotation_quaternion"):
                    flag = 1
                
                else:
                    print(fc.data_path)
                    
                if flag < 0:
                    continue
                    
                for key in fc.keyframe_points:
                    anim.add_value(key.co[0], bone_idx, flag, key.co[1])
                    
            self.anim_arr.append(anim)
                
    def load(self, obj, arm):
        if obj is None:
            pass
        
        self.mobj = obj
        self.marm = arm

        self.obj_verts = self.mobj.data.vertices
        self.obj_group_names = [g.name for g in self.mobj.vertex_groups]

        # Change to object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        
        # Load the mesh
        self.load_mesh()
        
        # Calculate the normal-vectors
        self.calc_normals()

        if arm is not None:
            # Change to object mode
            bpy.ops.object.mode_set(mode="POSE")

            # Load the bones
            self.load_bones()
        
            # Calculate the object-matrices for the joints
            self.calc_joint_matrices()
        
            # Link the bones to the vertices
            self.link_bones()
        
            # Add the joints and weights to the index-array
            self.fill_joints()
        
            # Load all animations
            self.load_animation()
        
    def write(self, path):
        of = open(path, "w")
        of.write("ao %s\n" % self.mobj.name)

        mbasis = np.array(self.marm.matrix_basis)
        print("Matrix", self.marm.name)
        print(mbasis)

        # Write vertices
        for v in self.vtx_arr:
            vtx = mbasis.dot(np.array([v.position.x, v.position.y, v.position.z, 1.0]))
            of.write("v %.4f %.4f %.4f\n" % (vtx[0], vtx[1], vtx[2]))

        # Write uv-coords
        for t in self.tex_arr:
            of.write("vt %.4f %.4f\n" % (t.x, t.y))

        # Write normals
        for n in self.nrm_arr:
            nrm = mbasis.dot(np.array([n[0], n[1], n[2], 1.0]))
            of.write("vn %.4f %.4f %.4f\n" % (nrm[0], nrm[1], nrm[2]))
            
        # Write vertex joints
        for j in self.jnt_arr:
            of.write("vj %d %d %d %d\n" % (j[0], j[1], j[2], j[3]))
            
        # Write vertex weights
        for w in self.wgt_arr:
            of.write("vw %.4f %.4f %.4f %.4f\n" % (w[0], w[1], w[2], w[3]))

        # Write indices
        tmp = 0
        of.write("f ")
        for i in self.idx_arr:
            for s in range(2, len(i), 1):
                of.write("%d" % (i[s] + 1))
                
                if s < len(i) - 1:
                    of.write("/")
                else:
                    of.write(" ")
            
            tmp += 1
            
            if tmp % 3 == 0:
                of.write("\n")
                
                if tmp < len(self.idx_arr) - 1:
                    of.write("f ")
                
        flg = 0        
            
        # Write the joints
        for b in self.bone_arr:
            parent = b.parent
            if parent >= 0:
                parent += 1
            of.write("j %s %d " % (b.name, parent))
            
            
            mat = b.matrix
            
            # Modify root bone
            if flg == 0:
                mat = mbasis @ mat
                flg = 1
            
            mat.transpose()
            for jv in mat:
                of.write("%.4f %.4f %.4f %.4f " % (jv[0], jv[1], jv[2], jv[3]))
                
            of.write("\n")

        # Write animations
        for a in self.anim_arr:
            length = a.keyframes[len(a.keyframes) - 1].timestamp
            of.write("a %s %d\n" % (a.name, int((length / 24) * 1000)))
            
            for k in a.keyframes:
                of.write("k %.4f\n" % (k.timestamp / length))
                        
                flg = 0
                
                for b in k.bones:
                    pos = b.loc
                    
                    if flg == 0:
                        loc =  self.marm.location
                        pos = np.array([pos[0] + loc[0], pos[1] + loc[1], pos[2] + loc[2]])
                        flg = 1
                    
                    of.write("ap %d %.4f %.4f %.4f\n" % (b.index + 1, pos[0], pos[1], pos[2]))

                flg = 0

                for b in k.bones:
                    rot = b.rot
                    
                    if flg == 0:
                        rot = np.array(q_mult(rot, self.marm.rotation_quaternion))
                        flg = 1
                        
                    of.write("ar %d %.4f %.4f %.4f %.4f\n" % (b.index + 1, rot[0], rot[1], rot[2], rot[3]))

        of.close()


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
        if i.type == "ARMATURE":
            arm = i
            break
        
    amo = AMOModel()
    amo.load(mdl, arm)
    amo.write(os.fsencode(filepath)) 
    return {'FINISHED'}


if __name__ == "__main__":
    register()