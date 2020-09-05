bl_info = {
    "name": "AmoAddon",
    "description": "Import and export models as dot-amo",
    "author": "VasallSoftware",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "File > Export > Dot-Amo",
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



def vtx_nrm(v):
    norm = np.linalg.norm(v, ord = 1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm



class amo_handler(bpy.types.Operator):
    """My Object Moving Script"""
    bl_idname = "export_scene.expoamo"       # Unique identifier for buttons and menu items to reference.
    bl_label = "Export dot-amo"        # Display name in the interface.
    bl_options = {"Present"}
    
    filename_ext    = ".amo";
         
    vtx_arr = []
    vtx_num = 0
    vtx_i = 0
    uv_arr = []
    nrm_arr = []
    idx_arr = []
    idx_num = 0
    idx_i = 0

    
    def __init__(self):
        pass
    
    def execute(self, context):
        # Change to object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        pass
    
    def load_data(self, objects):
        for obj in objects:
            mesh = obj.data

            # Collect and store the indices
            self.idx_i = 0
            for f in mesh.polygons:
                self.idx_arr.append([f.index, 0, f.vertices[0] + self.vtx_num])
                self.idx_arr.append([f.index, 1, f.vertices[1] + self.vtx_num])
                self.idx_arr.append([f.index, 2, f.vertices[2] + self.vtx_num])
                self.idx_i += 3

            # Collect and store all vertices
            self.vtx_i = 0
            for i in range(len(mesh.vertices)):
                print(obj.matrix_world)
                # Convert position to world space
                v = obj.matrix_world @ obj.data.vertices[i].co
                self.vtx_arr.append(v)
                self.vtx_i += 1
             
            # Collect and store the UV coordinates
            same = -1
            for i in range(self.idx_i):
                tmp = self.idx_num + i
                same = -1
            
                uv = mesh.uv_layers.active.data[self.idx_arr[tmp][0] * 3 + self.idx_arr[tmp][1]].uv
            
                for j in range(len(self.uv_arr)):
                    if(self.uv_arr[j] == uv):
                        same = j
                        break;
                
                if(same < 0):
                    self.uv_arr.append(uv)
                    self.idx_arr[tmp].append(len(self.uv_arr))
                
                else:
                    self.idx_arr[tmp].append(same + 1)


            # Calculate normal vectors
            same = -1
            for v in range(0, idx_i, 3):
                tmp = self.idx_num + v
                arr = [self.idx_arr[tmp][2], self.idx_arr[tmp + 1][2], self.idx_arr[tmp + 2][2]]
            
                a = self.vtx_arr[arr[0]] - self.vtx_arr[arr[1]]
                b = self.vtx_arr[arr[2]] - self.vtx_arr[arr[1]]
            
                nrm = obj.matrix_world @ np.cross(a, b)
                nrm = vec_nrm(nrm)
            
                same = -1
                for i in range(len(self.nrm_arr)):
                    if((nrm == self.nrm_arr[i]).all()):
                        same = i
                        break
                
                if same < 0:
                    self.nrm_arr.append(nrm)
                
                    idx = len(nrm_arr)
                    self.idx_arr[tmp].append(idx)
                    self.idx_arr[tmp + 1].append(idx)
                    self.idx_arr[tmp + 2].append(idx)
                
                else:
                    self.idx_arr[tmp].append(same + 1)
                    self.idx_arr[tmp + 1].append(same + 1)
                    self.idx_arr[tmp + 2].append(same + 1)
                    
            self.idx_num += self.idx_i
            self.idx_i = 0
            
            self.vtx_num += self.vtx_i
            self.vtx_i = 0
            
    def write_file(self):
        of = open("test.obj", "w")
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
        
def register():
    pass
    
def unregister():
    pass
        
        
if __name__ == "__main__":
    register()