bl_info = {
    "name": "HNDAddon",
    "description": "Import and export models as handheld dot-amo-hnd",
    "author": "clusterwerk",
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
from transforms3d.quaternions import quat2mat

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


class ExportHNDAMO(bpy.types.Operator, ExportHelper):
    """Save objects in HND-format"""
    bl_idname = "export_mesh.hnd"
    bl_label = "Export Dot-HND"
    filter_glob = StringProperty(
        default="*.hnd",
        options={'HIDDEN'},
    )
    check_extension = True
    filename_ext = ".hnd"

    def execute(self, context):
        return save(self, context, self.filepath)


def menu_func_export(self, context):
    self.layout.operator(ExportHNDAMO.bl_idname, text="Dot-HND-AMO (.hnd)")


classes = (
    ExportHNDAMO,
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
    
class Hook:
    def __init__(self, idx, vec):
        self.idx = idx
        self.vec = vec
    
class Handheld:
    def __init__(self, mdl, brl, hooks):
        # Get the name of the handheld
        name_tmp = mdl.name.split("_")
        self.name = name_tmp[len(name_tmp) - 1]
        
        # Get the index of the parent-hook
        name_tmp = mdl.parent.name.split("_")
        self.par_hook = int(name_tmp[len(name_tmp) - 1])
        
        self.hook_arr = []
        
        for h in hooks:
            # Calculate the vector from the handheld to the weapon
            name_tmp = h.name.split("_")
            idx = int(name_tmp[len(name_tmp) - 1])
            delta = mdl.location - h.location
            self.hook_arr.append(Hook(idx, delta))
            
            # Calculate the offset of the barrel
            if idx == self.par_hook:
                self.brl_off = brl.location - h.location
                
        self.hook_arr.sort(key=lambda x: x.idx, reverse=False)
        
        self.vtx_arr = []
        self.tex_arr = []
        self.nrm_arr = []
        self.idx_arr = []
        
        # Load the model-mesh of the handheld
        self.loadMesh(mdl)
        
        # Calculate the normals of the model
        self.calcNormals()
    
    def loadMesh(self, obj):
        mesh = obj.data
        
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
            mat = obj.matrix_basis.copy()
            
            mat[0][3] = 0
            mat[1][3] = 0
            mat[2][3] = 0
            
            v = mat @ mesh.vertices[i].co
            
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

    def calcNormals(self):
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
    
    def write(self, path):
        of = open(path, "w")
        
        # Write the name of the handheld
        of.write("hnd %s\n" % self.name)
        
        # Write the index of the parent-hook
        of.write("phk %d\n" % self.par_hook)
        
        # Write the vectors from the hooks to the handheld
        for i in self.hook_arr:
            of.write("hok %d " % i.idx)
            
            for j in range(0, 3, 1):
                of.write("%.4f " % i.vec[j])
                
            of.write("\n")
        
        # Write the position of the aiming-barrel
        of.write("bof ")
        for i in range(0, 3, 1):
            of.write("%.4f " % self.brl_off[i])
        of.write("\n")
        
        of.write("end\n")
        
        
        of.write("ao %s %d\n" % (self.name, 1))
        
        # Write vertices
        of.write("v %d\n" % len(self.vtx_arr))
        for v in self.vtx_arr:
            vtx = np.array([v.position.x, v.position.y, v.position.z, 1.0])
            of.write("%.4f %.4f %.4f\n" % (vtx[0], vtx[1], vtx[2]))

        # Write uv-coords
        of.write("vt %d\n" % len(self.tex_arr))
        for t in self.tex_arr:
            of.write("%.4f %.4f\n" % (t.x, t.y))

        mbasis = np.identity(4)
        # Write normals
        of.write("vn %d\n" % len(self.nrm_arr))
        for n in self.nrm_arr:
            nrm = mbasis.dot(np.array([n[0], n[1], n[2], 1.0]))
            of.write("%.4f %.4f %.4f\n" % (nrm[0], nrm[1], nrm[2]))
        
        # Write indices
        tmp = 0
        of.write("f %d\n" % (len(self.idx_arr) / 3))
        for i in self.idx_arr:
            for s in range(2, len(i), 1):
                of.write("%d " % (i[s]))
                
            tmp += 1
            
            if tmp % 3 == 0:
                of.write("\n")
        
        of.write("end\n")
        of.close()
        
    
def save(operator, context, filepath):
    # Get all selected objects
    objects = context.selected_objects
    
    mdl = None
    hooks = []
    barrel = None
    
    for i in objects:
        if i.type == "MESH":
            if i.name.find("hh_") != -1:
                hooks.append(i)
            elif i.name.find("hndb_") != -1:
                barrel = i
            elif i.name.find("hnd_") != -1:
                mdl = i
                
                
    if mdl is None:
        print("No model selected")
        return {'FINISHED'}
    
    if len(hooks) <= 0:
        print("No hooks selected")
        return {'FINISHED'}
    
    if barrel is None:
        print("No barrel selected")
        return {'FINISHED'}
    
    hnd = Handheld(mdl, barrel, hooks)
    hnd.write(os.fsencode(filepath))
    
    return {'FINISHED'}
    
    
if __name__ == "__main__":
    register()