from __future__ import division, print_function
import unittest
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, LinearBlockGS, PetscKSP
from openmdao.utils.assert_utils import assert_rel_error

class Test(unittest.TestCase):

    def test(self):

        # Create a dictionary to store options about the surface
        mesh_dict = {'num_y' : 7,
                     'num_x' : 2,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surface = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'type' : 'aero',
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'S_ref_type' : 'wetted', # how we compute the wing area,
                                             # can be 'wetted' or 'projected'
                    'fem_model_type' : 'tube',

                    'twist_cp' : twist_cp,
                    'mesh' : mesh,
                    'num_x' : mesh.shape[0],
                    'num_y' : mesh.shape[1],

                    # Aerodynamic performance of the lifting surface at
                    # an angle of attack of 0 (alpha=0).
                    # These CL0 and CD0 values are added to the CL and CD
                    # obtained from aerodynamic analysis of the surface to get
                    # the total CL and CD.
                    # These CL0 and CD0 values do not vary wrt alpha.
                    'CL0' : 0.0,            # CL of the surface at alpha=0
                    'CD0' : 0.015,            # CD of the surface at alpha=0

                    # Airfoil properties for viscous drag calculation
                    'k_lam' : 0.05,         # percentage of chord with laminar
                                            # flow, used for viscous drag
                    't_over_c' : 0.15,      # thickness over chord ratio (NACA0015)
                    'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                            # thickness
                    'with_viscous' : True,  # if true, compute viscous drag
                    }

        surfaces = [surface]

        # Create the problem and the model group
        prob = Problem()

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v', val=248.136, units='m/s')
        indep_var_comp.add_output('alpha', val=5.)
        indep_var_comp.add_output('M', val=0.84)
        indep_var_comp.add_output('re', val=1.e6, units='1/m')
        indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
        indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')

        prob.model.add_subsystem('prob_vars',
            indep_var_comp,
            promotes=['*'])

        geom_group = Geometry(surface=surface)

        # Add tmp_group to the problem as the name of the surface.
        # Note that is a group and performance group for each
        # individual surface.
        prob.model.add_subsystem(surface['name'], geom_group)


        # Create the aero point group and add it to the model
        aero_group = AeroPoint(surfaces=surfaces)
        point_name = 'aero_point_0'
        prob.model.add_subsystem(point_name, aero_group,
            promotes_inputs=['v', 'alpha', 'M', 're', 'rho', 'cg'])

        name = surface['name']

        # Connect the mesh from the geometry component to the analysis point
        prob.model.connect(name + '.mesh', point_name + '.' + name + '.def_mesh')

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        prob.model.connect(name + '.mesh', point_name + '.aero_states.' + name + '_def_mesh')

        from openmdao.api import ScipyOptimizeDriver
        prob.driver = ScipyOptimizeDriver()

        # # Setup problem and add design variables, constraint, and objective
        prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
        prob.model.add_constraint(point_name + '.wing_perf.CL', equals=0.5)
        prob.model.add_objective(point_name + '.wing_perf.CD', scaler=1e4)

        # Set up the problem
        prob.setup()

        prob.run_driver()

        assert_rel_error(self, prob['aero_point_0.wing_perf.CD'][0], 0.033389699871650073, 1e-8)
        assert_rel_error(self, prob['aero_point_0.wing_perf.CL'][0], 0.5, 1e-8)
        assert_rel_error(self, prob['aero_point_0.CM'][1], -0.18451822790794759, 1e-8)


if __name__ == '__main__':
    unittest.main()