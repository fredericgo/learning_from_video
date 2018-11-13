from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile
import os

class J3dPlotter(object):
	def __init__():
		pass

	def plot(joints3d):
    	joints3d = joints3d[:, :14, :]
    	tempdir = tempfile.mkdtemp()
    	print(tempdir)
	    for i, x in enumerate(joints3d):
	        outfilename = os.path.join(tempdir, 'temp_{:05d}.png'.format(i))
	    	self._p3d(x, outfilename)

	def _p3d(self, joints, filename):
		limb_parents = [1, 2, 12, 12, 3, 4, 7, 8, 12, 12, 9, 10, 13, 13, 13]

		fig = Figure()
		ax = fig.add_subplot(111, projection='3d')
	    ax.set_xlim(0,2)
	    ax.set_ylim(0,2)
	    ax.set_zlim(0,10)
	    ax.view_init(elev=-70,azim=-90)
	    ax.scatter3D(joints[:,0],
	              	 joints[:,1],
	                 joints[:,2], 'gray')

	    for i in range(joints3d.shape[0]):
	        x_pair = [joints[i, 0], joints[limb_parents[i], 0]]
	        y_pair = [joints[i, 1], joints[limb_parents[i], 1]]
	        z_pair = [joints[i, 2], joints[limb_parents[i], 2]]
	        ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
	    ax.axis('off')
	    canvas = FigureCanvasAgg(fig)
	    canvas.print_figure(filename)
