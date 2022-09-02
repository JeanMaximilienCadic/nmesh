from math import acos, pi
import trimesh
from gnutools.utils import id_generator
from scipy.spatial.distance import cdist as euclidean_distances
from .functional import *
from PIL import Image
import random


class NMesh(trimesh.Trimesh):
    def __init__(self, filename=None, mesh=None, list=None, *args, **kwargs):
        """
        Initialize a mesh: Load a filename or copy an existing mesh

        :param filename: path to the mesh to load
        :param mesh: NMesh
        """
        super(trimesh.Trimesh, self).__init__()
        self.load(filename=filename, mesh=mesh, list=list)

    def __getattr__(self, name):
        """
        Call parent methods.

        :param name:
        :return:
        """
        try:
            return getattr(self.mesh, name)
        except AttributeError:
            raise AttributeError("Child' object has no attribute '%s'" % name)

    def load(self, filename=None, mesh=None, list=None):
        """

        :param filename:
        :param mesh:
        :param list:
        :return:
        """
        if filename is not None:
            self.mesh = trimesh.load(filename)
            self.filename = filename
        elif list is not None:
            self.mesh = list[0]
            for m in list[1:]:
                self.mesh += m
        else:
            self.mesh = mesh

    def components(
        self, only_watertight=False, r=None, min_vertices=-1, max_vertices=None
    ):
        """
        Return the number of disconnected components and filter by the size of splited meshes

        :param only_watertight:
        :param min_vertices:
        :return:
        """
        meshes = (
            [NMesh(mesh=m)
             for m in self.split(only_watertight=only_watertight)]
            if r is None
            else [
                NMesh(mesh=m)
                for m in self.split(only_watertight=only_watertight)
                if len(m.vertices) in r
            ]
        )
        # Filter by size
        if max_vertices is not None:
            return [
                m
                for m in meshes
                if len(m.vertices) in range(min_vertices, max_vertices)
            ]
        else:
            return [m for m in meshes if len(m.vertices) >= min_vertices]

    def crop_bounding_box(self, r):
        """
        Reduce a mesh to a specific region in the space

        :param r: region of reference to crop the mesh
        :return:
        """

        inds_vertices = np.argwhere(
            (np.min(self.vertices - r[0], axis=1) >= 0)
            & (np.max(self.vertices - r[1], axis=1) <= 0)
        ).reshape(
            -1,
        )
        self.vertices_subset(inds=inds_vertices)

    def vertices_subset(self, inds):
        """
        Update the vertices with a subset.

        :param inds:
        :return:
        """
        inds_faces = np.argwhere(
            np.in1d(self.faces[:, 0], inds)
            & np.in1d(self.faces[:, 1], inds)
            & np.in1d(self.faces[:, 2], inds)
        ).reshape(
            -1,
        )

        table = dict([(value, key) for key, value in enumerate(inds)])

        self.faces = self.faces[inds_faces]
        try:
            if not len(self.faces) == len(self.visual.face_colors):
                self.visual.face_colors = self.visual.face_colors[inds_faces]
        except:
            pass
        self.vertices = self.vertices[inds]
        try:
            if not len(self.vertices) == len(self.visual.vertex_colors):
                self.visual.vertex_colors = self.visual.vertex_colors[inds]
        except:
            pass
        self.faces = [[table[v] for v in f] for f in self.faces]

    def compress(self, dims):
        """
        Compress the mesh

        :param dims:
        :return:
        """
        T = np.mean(self.ranges(), axis=0)
        self.translate(translation=-T)
        L = self.length()
        self.vertices /= L
        self.vertices = self.vertices * dims[0]
        self.vertices = self.vertices.astype(int)

    def uncompress(self):
        """
        Uncompress the mesh regarding 1/N factor

        :return:
        """
        if self._compressed:
            # Restore
            self.faces = self._faces
            self.vertices = self._vertices
            self.visual.vertex_colors = self._vertex_colors
            self._compressed = False

    def rotate(self, theta, axis_rotation):
        """
        Rotate the mesh around the origin axis

        :param theta:
        :param axis_rotation:
        :return:
        """
        self.vertices = rotate(
            vertices=self.vertices, theta=theta, axis_rotation=axis_rotation
        )

    def rgb(self, res=64, neighbors=[], opacity=1):
        """
        Set a color to a mesh

        :param res:
        :param neighbors:
        :param opacity:
        :return:
        """
        # Bounding box and compression
        vertices = np.unique(np.array(self.vertices * 10, dtype=int), axis=0)
        r = ranges(vertices)
        l = length(vertices)
        r_extend = np.ceil(
            np.array([[-max(r[1])] * 3, [max(r[1])] * 3]) * 1.28)
        for i, _ in enumerate(np.array(neighbors)[:, 0]):
            neighbors[i][0].vertices = bounding_box(
                vertices=np.unique(
                    np.array(neighbors[i][0].vertices * 10, dtype=int), axis=0
                ),
                r=r_extend,
            )

        # Translate
        vertices = translate(vertices, -r_extend[0])
        mshape = ranges(vertices)[1][1:]
        for i, _ in enumerate(np.array(neighbors)[:, 0]):
            neighbors[i][0].vertices = translate(
                np.array(neighbors[i][0].vertices, dtype=int), -r_extend[0]
            )
            mshape = [
                int(max(mshape[0], ranges(neighbors[i][0].vertices)[1][1])),
                int(max(mshape[1], ranges(neighbors[i][0].vertices)[1][2])),
            ]

        # Prepare the image
        img = np.zeros((mshape[0] + 1, mshape[1] + 1, 3))
        for channel, (neighbor, op) in enumerate(neighbors):
            for x, y, z in tuple(neighbor.vertices):
                img[int(y), int(z), channel + 1] = max(
                    op * int(x), img[int(y), int(z), channel + 1]
                )

        for x, y, z in tuple(vertices):
            img[int(y), int(z), 0] = max(
                opacity * int(x), img[int(y), int(z), 0])

        mchannel = np.max(img)
        img /= mchannel
        img *= 255

        img = cv2.rotate(np.array(img, dtype=np.uint8),
                         cv2.ROTATE_90_COUNTERCLOCKWISE)
        for _ in range(7):
            for channel in range(3):
                img[:, :, channel] = ndimage.maximum_filter(
                    img[:, :, channel], 2)
        img = cv2.resize(img, dsize=(res, res))
        return img

    def ranges(self):
        """
        Find the bounding box of the mesh

        :return:
        """
        return np.array(
            [
                [self.vertices[:, k].min() for k in range(3)],
                [self.vertices[:, k].max() for k in range(3)],
            ]
        )

    def length(self):
        """
        Find the length of a mesh

        :return:
        """
        return length(self.vertices)

    def translate(
        self, translation, axis=None, positive=False, negative=False, inds=None
    ):
        """
        Move the mesh in the space

        :param translation:
        :param axis:
        :param positive:
        :param negative:
        :param inds:
        :return:
        """
        if positive:
            [vertices, trans] = center(copy.deepcopy(self.vertices()))
            vertices[np.where(vertices[:, axis] >= 0)] += translation
            vertices = translate(vertices, -trans)
            self.vertices(vertices)
        elif negative:
            [vertices, trans] = center(copy.deepcopy(self.vertices()))
            vertices[np.where(vertices[:, axis] < 0)] += translation
            vertices = translate(vertices, -trans)
            self.vertices(vertices)
        else:
            if inds is None:
                self.mesh.vertices += translation
            else:
                self.mesh.vertices[inds] += translation

    def set_color(self, c):
        """
        Set the color of the mesh

        :param c:
        :return:
        """
        self.visual.face_colors = c

    def colorize_components(self, r=None):
        """
        Set a random color to each of the components

        :param r:
        :return:
        """
        splits_all = self.components()
        splits = [s for s in splits_all if len(s.vertices) in r]
        [s.set_color(random_color()) for s in splits]
        self.load(list=list(splits) +
                  list(s for s in splits_all if not s in splits))

    def meshlab(self, script_name, ext_in="ply", ext_out="ply"):
        """
        Apply a filter to the mesh

        :param name:
        :return:
        """

        script = f"__data__/mlx/{script_name}.mlx"
        ply_file = "/tmp/{}".format(id_generator())
        file_in = "{}.{}".format(ply_file, ext_in)
        file_out = "{}/{}.{}".format(parent(file_in), name(file_in), ext_out)
        self.export(file_in)
        command = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i "{}" -o "{}" -s {} -om vc'.format(
            file_in, file_out, script
        )
        os.system(command)
        print(">> meshlab : {}".format(command))
        self.load("{}.{}".format(ply_file, ext_out))
        os.system("rm {}.{}".format(ply_file, ext_out))

    def filter(self, colormin=0.0, colormax=1.0):
        """
        Filter faces by color

        :param colormin:
        :param colormax:
        :return:
        """
        fcolors = np.array([rgb2flaot(rgb)
                           for rgb in self.visual.face_colors[:, :3]])
        inds = np.argwhere((fcolors >= colormin) & (fcolors <= colormax)).reshape(
            -1,
        )
        self.visual.face_colors = self.visual.face_colors[inds]
        self.faces = self.faces[inds]

    def main_component(self):
        """
        Return the main component in a mesh.

        :return:
        """
        candidates = np.array(
            [(cmpt, len(cmpt.vertices)) for cmpt in self.components()]
        )
        amax = np.argmax(candidates[:, 1])
        return candidates[amax, 0]

    def origin(self, T):
        """
        Set the origin of the mesh

        :param T:
        :return:
        """
        T0 = -self.ranges()[0]
        self.translate(T0 + T)
        return T0

    def matrix3d(self, dim=64):
        """
        Convert the mesh to a 3d matrix

        :return:
        """
        T0 = self.origin([0, 0, 0])
        uvertices = np.unique(self.vertices, axis=0).astype(int)
        assert np.min(uvertices) >= 0
        assert np.max(uvertices) < dim
        M = np.zeros((dim, dim, dim, 1))
        M[uvertices[:, 0], uvertices[:, 1], uvertices[:, 2]] = 1
        return M

    def inter_bbox(self, cmpt):
        """

        :param cmpt:
        :return:
        """
        min_ref, max_ref = self.ranges()
        min_cmpt, max_cmpt = cmpt.ranges()
        condition = (min_cmpt <= max_ref) & (max_cmpt >= min_ref)
        return np.min(condition)

    def in_bbox(self, cmpt):
        bbox = cmpt.ranges()
        return len(crop_bounding_box(self.vertices, bbox)) > 0

    def add_vertice(self, v):
        """
        Concatenate vertices.

        :param v:
        :return:
        """
        vertices = list(self.vertices)
        vertices.append(v)
        self.vertices = vertices

    def shot(self, resolution=[400, 400]):
        """

        :return:
        """
        img_path = "/tmp/{}.png".format(id_generator())
        scene = self.scene()
        # saving an image requires an opengl context, so if -nw
        # is passed don't save the image
        try:
            # increment the file name
            # save a render of the object as a png
            png = scene.save_image(resolution=resolution, visible=True)
            with open(img_path, "wb") as f:
                f.write(png)
                f.close()
            img = cv2.imread(img_path)
            os.system("rm {}".format(img_path))
            return img
        except BaseException as E:
            print("unable to save image", str(E))

    def to_pmeshlab(self):
        from pymeshlab import Mesh as _Mesh
        from pmeshlab import Mesh

        return Mesh(
            input=_Mesh(
                vertex_matrix=self.vertices,
                face_matrix=self.faces,
                v_normals_matrix=self.vertex_normals,
                f_normals_matrix=self.face_normals,
                f_color_matrix=self.visual.face_colors,
                v_color_matrix=self.visual.vertex_colors,
            )
        )

    def discrete_curvatures(self):
        return self.to_pmeshlab().discrete_curvatures()[0].to_NMesh()

    def filter_face_colors(self, color, threshold=0):
        m = NMesh(self.copy())
        condition = np.linalg.norm(
            m.visual.face_colors - color, axis=1) < threshold
        finds = np.argwhere(condition).flatten()
        vinds = np.unique(m.faces[finds])
        m.vertices_subset(vinds)
        return m

    def random_shots(self, niters=4):
        imgs = []
        for _ in range(niters):
            self.rotate(axis_rotation=random.randint(
                0, 3), theta=pi * random.random())
            imgs.append(self.shot())
        imgs = np.array(imgs)
        return Image.fromarray(imgs.reshape(400 * niters, 400, 3).swapaxes(0, 1))
