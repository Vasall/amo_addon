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
from transforms3d.quaternions import quat2mat
import bpy
import numpy as np
import math
import os
bl_info = {
    "name": "AmoAddon",
    "description": "Import and export models as dot-amo",
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



def vec_nrm(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


# The following code is used to rotate a vector by a quaternion.
# See https://stackoverflow.com/a/4870905 for more info.
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]



# A class containing a joint and the weight, the vertex is affected by.
# This is necessary as the joint-data is read entry by entry and new vertex-joints,
# have to be added dynamically.
class VertexJointEntry:
    def __init__(self, jnt, wgt):
        self.joint = jnt
        self.weight = wgt

# A class containing all data of a single vertex.
class Vertex:
    def __init__(self, idx, pos):
        self.index = idx
        self.position = pos
        self.jointsArray = []

        self.joints = []
        self.weights = []

    def add_joint(self, joint, weight):
        self.jointsArray.append(VertexJointEntry(joint, weight))

    def sort_joints(self):
        self.jointsArray.sort(key=lambda x: x.weight, reverse=True)

        for i in range(0, len(self.jointsArray), 1):
            self.joints.append(self.jointsArray[i].joint)
            self.weights.append(self.jointsArray[i].weight)


class Joint:
    def __init__(self, index, name, parent):
        self.index = index
        self.name = name
        self.parent = parent
        self.matrix_loc = []
        self.matrix_rel = []


class JointKeyframe:
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
        self.joints = []

    def add_key(self, bone, flag, value):
        found = -1
        for b in range(len(self.joints)):
            if self.joints[b].index == bone:
                found = b
                break

        if found >= 0:
            if flag == 0:
                self.joints[found].add_loc(value)
            elif flag == 1:
                self.joints[found].add_rot(value)

        else:
            key = JointKeyframe(bone)

            if flag == 0:
                key.add_loc(value)
            elif flag == 1:
                key.add_rot(value)

            self.joints.append(key)


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

    def cleanup(self):
        pass


class PosScl:
    def __init__(self, pos, scl):
        self.pos = pos
        self.scl = scl


class RigBox:
    #
    # @idx: The index of the parent-bone
    # @pos: The position of the box relative to the armature
    # @mat: The rotation matrix of the box
    #
    def __init__(self, idx, pos, scl, mat):
        self.idx = idx

        self.pos = pos
        self.scl = scl

        self.mat = np.asmatrix(mat)


class ItemHook:
    #
    #
    #
    def __init__(self, idx, jnt_idx, pos, dir, mat):
        self.idx = idx
        self.jnt_idx = jnt_idx
        self.pos = pos
        self.dir = dir
        self.mat = mat
        
        print("Matrix")
        print(self.mat)


class AMOModel:
    def __init__(self):
        self.mobj = None
        self.marm = None
        self.data_m = 0

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

        self.hk_arr = []

        self.cflg = 0
        self.bbc = None
        self.nec = None
        self.cm_vtx_arr = []
        self.cm_idx_arr = []
        self.rb_arr = []

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

            tex = mesh.uv_layers.active.data[self.idx_arr[i]
                                             [0] * 3 + self.idx_arr[i][1]].uv

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
            arr = [self.idx_arr[v][2], self.idx_arr[v + 1]
                   [2], self.idx_arr[v + 2][2]]

            a = self.vtx_arr[arr[1]].position - self.vtx_arr[arr[0]].position
            b = self.vtx_arr[arr[1]].position - self.vtx_arr[arr[2]].position

            # TODO add world matrix
            nrm = np.cross(a, b)
            nrm = vec_nrm(nrm)
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

        if len(self.bone_arr) > 0:
            self.data_m = self.data_m | (1 << 1)

    def get_bone(self, name):
        for b in self.bone_arr:
            if b.name == name:
                return b.index

        return -1

    def link_bones(self):
        for bone in self.marm.pose.bones:
            bidx = self.get_bone(bone.name)

            gidx = self.mobj.vertex_groups[bone.name].index

            bone_verts = [v for v in self.obj_verts if gidx in [
                g.group for g in v.groups]]

            for v in bone_verts:
                for g in v.groups:
                    if g.group == gidx:
                        self.vtx_arr[v.index].add_joint(bidx, g.weight)

        for i in range(0, len(self.vtx_arr), 1):
            self.vtx_arr[i].sort_joints()

    def calc_joint_matrices(self):
        for b in self.bone_arr:
            b.matrix_loc = self.marm.matrix_world @ self.marm.data.bones[b.name].matrix_local

        for b in self.bone_arr:
            mat = b.matrix_loc

            if b.parent > -1:
                inv = np.linalg.inv(self.bone_arr[b.parent].matrix_loc)
                mat = inv @ mat

            b.matrix_rel = mat

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
        # Check if model has bones
        if len(self.bone_arr) < 1:
            pass

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
                    a = 0
                    #print(fc.data_path)

                if flag < 0:
                    continue

                for key in fc.keyframe_points:
                    anim.add_value(key.co[0], bone_idx, flag, key.co[1])

            self.anim_arr.append(anim)

        # Cleanup the animation data to remove redundant keyframes
        for a in self.anim_arr:
            a.cleanup()

        if len(self.anim_arr) > 0:
            self.data_m = self.data_m | (1 << 2)

    def loadMesh(self, obj):
        self.mobj = obj

        self.obj_verts = self.mobj.data.vertices
        self.obj_group_names = [g.name for g in self.mobj.vertex_groups]

        # Change to object mode
        bpy.ops.object.mode_set(mode="OBJECT")

        # Load the mesh
        self.load_mesh()

        # Calculate the normal-vectors
        self.calc_normals()

        # Update data-mask
        self.data_m = self.data_m | (1 << 0)

    def loadRigAnim(self, arm):
        self.marm = arm

        # Select the armature
        bpy.data.objects[arm.name].select_set(True)

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

    #
    # HOOKS
    #

    def loadHooks(self, arr):
        for o in arr:
            txt = o.name.split("_")
            idx = int(txt[len(txt) - 1])
            jnt_idx = self.get_bone(o.parent_bone)
            pos = o.matrix_world.to_translation()

            print("Base Matrix")
            print(o.matrix_basis)
            
            print("World Matrix")
            print(o.matrix_world)

            dir = qv_mult(o.matrix_world.to_quaternion(), (0, 1, 0))
            dir = vec_nrm(dir)

            mat_world = np.asarray(o.matrix_world)
            
            
            print("Before Hook World Matrix:")
            print(mat_world)
            
            for k in range(0, 3, 1):
                mat_world[0][k] = mat_world[0][k] / o.scale[0]
                mat_world[1][k] = mat_world[1][k] / o.scale[1]
                mat_world[2][k] = mat_world[2][k] / o.scale[2]
            
            mat_inv_jnt = np.asarray(np.linalg.inv(self.bone_arr[jnt_idx].matrix_loc))
            mat = mat_inv_jnt @ mat_world
            
            print("Parent Join %d" % jnt_idx)
            
            print("Hook World Matrix:")
            print(mat_world)
            
            print("Joint Matrix")
            print(self.bone_arr[jnt_idx].matrix_loc)
            
            print("Inverse Joint Matrix:")
            print(mat_inv_jnt)
            
            print("Resulting Matrix")
            print(mat)

            self.hk_arr.append(ItemHook(idx, jnt_idx, pos, dir, mat))

        self.hk_arr.sort(key=lambda x: x.idx, reverse=False)
        self.data_m = self.data_m | (1 << 3)

    #
    # COLLISION_HANDLES
    #

    def loadBBCollision(self, obj):
        self.bbc = PosScl(obj.location, obj.scale)
        self.data_m = self.data_m | (1 << 10)

    def loadNECollision(self, obj):
        self.nec = PosScl(obj.location, obj.scale)
        self.data_m = self.data_m | (1 << 11)

    def loadCMCollision(self, obj):
        mesh = obj.data

        if len(mesh.polygons[0].vertices) == 3:
            for f in mesh.polygons:
                self.cm_idx_arr.append([f.index, 0, f.vertices[0]])
                self.cm_idx_arr.append([f.index, 1, f.vertices[1]])
                self.cm_idx_arr.append([f.index, 2, f.vertices[2]])
        else:
            for f in mesh.polygons:
                self.cm_idx_arr.append([f.index, 0, f.vertices[3]])
                self.cm_idx_arr.append([f.index, 1, f.vertices[0]])
                self.cm_idx_arr.append([f.index, 2, f.vertices[1]])

            for f in mesh.polygons:
                self.cm_idx_arr.append([f.index, 0, f.vertices[3]])
                self.cm_idx_arr.append([f.index, 1, f.vertices[1]])
                self.cm_idx_arr.append([f.index, 2, f.vertices[2]])

        # Collect and store all vertices
        for i in range(len(mesh.vertices)):
            # Convert position to world space
            v = obj.matrix_world @ mesh.vertices[i].co
            self.cm_vtx_arr.append(Vertex(len(self.vtx_arr), v))

        self.data_m = self.data_m | (1 << 12)

    def loadRBCollision(self, arr):
        for o in arr:
            idx = self.get_bone(o.parent_bone)
            pos = o.location
            scl = o.scale
            mat = np.asmatrix(o.matrix_basis)

            #mat.itemset(3,  0)
            #mat.itemset(7,  0)
            #mat.itemset(11, 0)

            self.rb_arr.append(RigBox(idx, pos, scl, mat))

        self.data_m = self.data_m | (1 << 13)

    def write(self, path):
        of = open(path, "w")
        of.write("ao %s %d\n" % (self.mobj.name, self.data_m))

        if self.marm is not None:
            mbasis = np.array(self.marm.matrix_basis)
        else:
            mbasis = np.identity(4)

        # Write vertices
        of.write("v %d\n" % len(self.vtx_arr))
        for v in self.vtx_arr:
            vtx = np.array([v.position.x, v.position.y, v.position.z, 1.0])
            of.write("%f %f %f\n" % (vtx[0], vtx[1], vtx[2]))

        # Write uv-coords
        of.write("vt %d\n" % len(self.tex_arr))
        for t in self.tex_arr:
            of.write("%f %f\n" % (t.x, t.y))

        # Write normals
        of.write("vn %d\n" % len(self.nrm_arr))
        for n in self.nrm_arr:
            nrm = mbasis.dot(np.array([n[0], n[1], n[2], 1.0]))
            of.write("%f %f %f\n" % (nrm[0], nrm[1], nrm[2]))

        # Write vertex joints
        if len(self.jnt_arr) > 0:
            of.write("vj %d\n" % len(self.jnt_arr))
            for j in self.jnt_arr:
                of.write("%d %d %d %d\n" % (j[0], j[1], j[2], j[3]))

        # Write vertex weights
        if len(self.wgt_arr) > 0:
            of.write("vw %d\n" % len(self.wgt_arr))
            for w in self.wgt_arr:
                # Normalize weights, so they add up to 1.0
                s = np.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + w[3] * w[3])

                vw = []

                for i in range(0, 4, 1):
                    vw.append(w[i] / s)

                of.write("%f %f %f %f\n" % (vw[0], vw[1], vw[2], vw[3]))

        # Write indices
        tmp = 0
        of.write("f %d\n" % (len(self.idx_arr) / 3))
        for i in self.idx_arr:
            for s in range(2, len(i), 1):
                of.write("%d " % i[s])
                
            tmp += 1
            
            if tmp % 3 == 0:
                of.write("\n")

        flg = 0

        # Write the joints
        if len(self.bone_arr) > 0:
            of.write("j %d\n" % len(self.bone_arr))
            for b in self.bone_arr:
                parent = b.parent
                of.write("%s %d " % (b.name, parent))

                mat = np.asmatrix(b.matrix_rel)
                mat.transpose()

                for i in range(0, 4, 1):
                    for j in range(0, 4, 1):
                        of.write("%f " % mat[j, i])

                of.write("\n")

        # Write animations
        if len(self.anim_arr) > 0:
            of.write("a %d\n" % len(self.anim_arr))
            for a in self.anim_arr:
                length = a.keyframes[len(a.keyframes) - 1].timestamp
                of.write("an %s %d %d\n" % (a.name, int((length / 24) * 1000), len(a.keyframes)))

                for k in a.keyframes:
                    of.write("k %f %d\n" % (k.timestamp / length, len(k.joints)))

                    # Writhe both the 
                    for b in k.joints:
                        of.write("%d " % b.index)
                        
                        pos = b.loc
                        of.write("%f %f %f " %
                                 (pos[0], pos[1], pos[2]))
                                 
                        rot = b.rot
                        of.write("%f %f %f %f\n" %
                                 (rot[0], rot[1], rot[2], rot[3]))

        #
        # Write Item-Hook
        #
        if len(self.hk_arr) > 0:
            of.write("hk %d\n" % len(self.hk_arr))
            for o in self.hk_arr:
                # Write the index of the item-hook
                of.write("%d " % o.idx)

                # Write the parent-bone
                of.write("%d " % o.jnt_idx)

                # Write the relative location to the armature
                of.write("%f %f %f " % (o.pos.x, o.pos.y, o.pos.z))

                # Write the forward-direction of the hook
                of.write("%f %f %f " % (o.dir[0], o.dir[1], o.dir[2]))
            
                mat = o.mat.transpose()  
                flat_mat = mat.flatten()

                for k in range(0, 16, 1):
                    of.write("%f " % flat_mat[k]);

                of.write("\n")

        #
        # Write collision-buffers
        #
        if self.bbc is not None:
            pos = self.bbc.pos
            scl = self.bbc.scl
            of.write("bb %f %f %f %f %f %f\n" %
                     (pos.x, pos.y, pos.z, scl.x, scl.y, scl.z))

        if self.nec is not None:
            pos = self.nec.pos
            scl = self.nec.scl
            of.write("ne %f %f %f %f %f %f\n" %
                     (pos.x, pos.y, pos.z, scl.x, scl.y, scl.z))

        if len(self.cm_vtx_arr) > 0:
            of.write("cv %d\n" % len(self.cm_vtx_arr))
            # Write vertices
            for v in self.cm_vtx_arr:
                vtx = np.array([v.position.x, v.position.y, v.position.z, 1.0])
                of.write("%f %f %f\n" % (vtx[0], vtx[1], vtx[2]))

            # Write indices
            tmp = 0
            of.write("ci %d\n" % (len(self.cm_idx_arr) / 3))
            for i in self.cm_idx_arr:
                of.write("%d " % i[2])

                tmp += 1

                if tmp % 3 == 0:
                    of.write("\n")

        # Write collision boxes
        if len(self.rb_arr) > 0:
            of.write("rb %d\n" % len(self.rb_arr))
            for o in self.rb_arr:
                # Write the index of the parent-bone
                of.write("%d " % o.idx)

                # Write position relative to armature
                for i in range(0, 3, 1):
                    of.write("%f " % o.pos[i])

                # Write scaling of the box
                for i in range(0, 3, 1):
                    of.write("%f " % o.scl[i])

                # Write the transformation-matrix
                for i in range(0, 4, 1):
                    for j in range(0, 4, 1):
                        of.write("%f " % o.mat[j, i])

                of.write("\n")

        of.write("end\n")
        of.close()


def save(operator, context, filepath):
    # Get all selected objects
    objects = context.selected_objects

    mdl_obj = None
    arm_obj = None
    bbc_obj = None
    nec_obj = None
    cmc_obj = None
    rbc_arr = []
    hk_arr = []

    for i in objects:
        if i.type == "MESH":
            if i.name.find("bb_") != -1:
                bbc_obj = i
            elif i.name.find("ne_") != -1:
                nec_obj = i
            elif i.name.find("cm_") != -1:
                cmc_obj = i
            elif i.name.find("rb_") != -1:
                rbc_arr.append(i)
            elif i.name.find("hh_") != -1:
                hk_arr.append(i)
            else:
                mdl_obj = i

        elif i.type == "ARMATURE":
            arm_obj = i

    amo = AMOModel()

    if mdl_obj is None:
        print("No model-object selected")
        return {"FINISHED"}

    amo.loadMesh(mdl_obj)

    if arm_obj is not None:
        amo.loadRigAnim(arm_obj)

    #
    # Load Hooks
    #
    if len(hk_arr) > 0:
        amo.loadHooks(hk_arr)

    #
    # Load collision-buffers
    #
    if bbc_obj is not None:
        amo.loadBBCollision(bbc_obj)

    if nec_obj is not None:
        amo.loadNECollision(nec_obj)

    if cmc_obj is not None:
        amo.loadCMCollision(cmc_obj)

    if len(rbc_arr) > 0:
        amo.loadRBCollision(rbc_arr)

    amo.write(os.fsencode(filepath))
    return {'FINISHED'}


if __name__ == "__main__":
    register()
