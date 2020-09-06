bl_info = {
    "name": "AmoAddon",
    "description": "Import and export models as dot-amo",
    "author": "VasallSoftware",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "File > Import-Export",
    "warning": "", # used for warning icon and text in addons panel
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/Scripts/My_Script",
    "tracker_url": "https://developer.blender.org/maniphest/task/edit/form/2/",
    "support": "COMMUNITY",
    "category": "Import-Export"
}


import bpy
import numpy as np
import math
import os
from bpy.props import (BoolProperty,
    FloatProperty,
    StringProperty,
    EnumProperty,
    )
from bpy_extras.io_utils import (ImportHelper,
    ExportHelper,
    unpack_list,
    unpack_face_list,
    axis_conversion,
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
    for c  in reversed(classes):
        bpy.utils.unregister_class(c)
    
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
        
       
       
def vtx_nrm(v):
    norm = np.linalg.norm(v, ord = 1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm
 
def save(operator, context, filepath):
    vtx_arr = []
    vtx_num = 0
    vtx_i = 0
    uv_arr = []
    nrm_arr = []
    idx_arr = []
    idx_num = 0
    idx_i = 0
    
    # Change to object mode
    bpy.ops.object.mode_set(mode="OBJECT")
    
    # Get all selected objects    
    objects = context.selected_objects
    
    for obj in objects:
        mesh = obj.data

        # Collect and store the indices
        idx_i = 0
        for f in mesh.polygons:
            idx_arr.append([f.index, 0, f.vertices[0] + vtx_num])
            idx_arr.append([f.index, 1, f.vertices[1] + vtx_num])
            idx_arr.append([f.index, 2, f.vertices[2] + vtx_num])
            idx_i += 3

        # Collect and store all vertices
        vtx_i = 0
        for i in range(len(mesh.vertices)):
            # Convert position to world space
            v = obj.matrix_world @ obj.data.vertices[i].co
            vtx_arr.append(v)
            vtx_i += 1
         
        # Collect and store the UV coordinates
        same = -1
        for i in range(idx_i):
            tmp = idx_num + i
            same = -1
        
            uv = mesh.uv_layers.active.data[idx_arr[tmp][0] * 3 + idx_arr[tmp][1]].uv
        
            for j in range(len(uv_arr)):
                if(uv_arr[j] == uv):
                    same = j
                    break;
            
            if(same < 0):
                uv_arr.append(uv)
                idx_arr[tmp].append(len(uv_arr))
            
            else:
                idx_arr[tmp].append(same + 1)


        # Calculate normal vectors
        same = -1
        for v in range(0, idx_i, 3):
            tmp = idx_num + v
            arr = [idx_arr[tmp][2], idx_arr[tmp + 1][2], idx_arr[tmp + 2][2]]
        
            a = vtx_arr[arr[0]] - vtx_arr[arr[1]]
            b = vtx_arr[arr[2]] - vtx_arr[arr[1]]
        
            # TODO add world matrix
            nrm = np.cross(a, b)
            nrm = vtx_nrm(nrm)
        
            same = -1
            for i in range(len(nrm_arr)):
                if((nrm == nrm_arr[i]).all()):
                    same = i
                    break
            
            if same < 0:
                nrm_arr.append(nrm)
            
                idx = len(nrm_arr)
                idx_arr[tmp].append(idx)
                idx_arr[tmp + 1].append(idx)
                idx_arr[tmp + 2].append(idx)
            
            else:
                idx_arr[tmp].append(same + 1)
                idx_arr[tmp + 1].append(same + 1)
                idx_arr[tmp + 2].append(same + 1)
                
        idx_num += idx_i
        idx_i = 0
        
        vtx_num += vtx_i
        vtx_i = 0
        
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
            of.write("%d/%d/%d " % (idx_arr[i][2] + 1, idx_arr[i][3], idx_arr[i][4]))
            of.write("%d/%d/%d " % (idx_arr[i + 1][2] + 1, idx_arr[i + 1][3], idx_arr[i + 1][4]))
            of.write("%d/%d/%d " % (idx_arr[i + 2][2] + 1, idx_arr[i + 2][3], idx_arr[i + 2][4]))
            of.write("\n")

        of.close()
        
        return {'FINISHED'}
    
    
        
if __name__ == "__main__":
    register()