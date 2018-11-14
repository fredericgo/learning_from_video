from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits import mplot3d
import tempfile
import os
import shutil

class J3dPlotter(object):
    def __init__(self):
        pass

    def plot(self, joints3d):
        joints3d = joints3d[:, :14, :]
        tempdir = tempfile.mkdtemp()
        print(tempdir)
        for i, x in enumerate(joints3d):
            outfilename = os.path.join(tempdir, 'temp_{:05d}.png'.format(i))
            self._p3d(x, outfilename)

        os.system("ffmpeg -i {}/temp_%5d.png -pix_fmt yuv420p -r 5 output.mp4".format(tempdir))
        shutil.rmtree(tempdir)
        

    def _p3d(self, joints, filename):
        parents = [1, 2, 12, 12, 3, 4, 7, 8, 12, 12, 9, 10, 13, 13, 13]
        #parents = [0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  
        #           9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        fig = Figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.set_xlim(0,2)
        #ax.set_ylim(0,2)
        #ax.set_zlim(0,10)
        ax.view_init(elev=90,azim=90)
        ax.scatter3D(joints[:,0],
                     joints[:,1],
                     joints[:,2], 'gray')

        for i in range(joints.shape[0]):
            x_pair = [joints[i, 0], joints[parents[i], 0]]
            y_pair = [joints[i, 1], joints[parents[i], 1]]
            z_pair = [joints[i, 2], joints[parents[i], 2]]
            ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
        ax.axis('off')
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
