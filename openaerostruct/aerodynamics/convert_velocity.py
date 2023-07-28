import numpy as np

import openmdao.api as om


class ConvertVelocity(om.ExplicitComponent):
    """
    Convert the freestream velocity magnitude into a velocity vector at each
    evaluation point. In this case, each of the panels sees the same velocity.
    This really just helps us set up the velocities for use in the VLM analysis.

    Parameters
    ----------
    alpha : float
        The angle of attack for the aircraft (all lifting surfaces) in degrees.
    beta : float
        The sideslip angle for the aircraft (all lifting surfaces) in degrees.
    v : float
        The freestream velocity magnitude.
    rotational_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.

    Returns
    -------
    freestream_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.
    """

    def initialize(self):
        self.options.declare("surfaces", types=list)
        self.options.declare(
            "rotational", False, types=bool, desc="Set to True to turn on support for computing angular velocities"
        )

    def setup(self):
        surfaces = self.options["surfaces"]
        rotational = self.options["rotational"]

        system_size = 0
        sizes = []

        # Loop through each surface and cumulatively add the number of panels
        # to obtain system_size.
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            size = (nx - 1) * (ny - 1)
            system_size += size
            sizes.append(size)

        self.system_size = system_size

        n = (ny-1)*(nx-1)
        rows = np.arange(0, 3*n, 1)
        cols = []
        for k in range(nx-1):
            for i in range(ny-1):
                for m in range(3):
                    cols.append(i)

        cols = np.array(cols)

        self.add_input("alpha", val=0.0, units="deg")
        self.add_input("beta", val=0.0, units="deg")
        self.add_input("v", val=1.0, units="m/s")
        self.add_input("velocity_distribution", shape=(ny-1), units="m/s")
        self.add_input("mesh", shape=(nx, ny, 3), units="m")

        if rotational:
            self.add_input("rotational_velocities", shape=(system_size, 3), units="m/s")

        self.add_output("freestream_velocities", shape=(system_size, 3), units="m/s")

        self.declare_partials("freestream_velocities", "alpha")
        self.declare_partials("freestream_velocities", "beta")
        self.declare_partials("freestream_velocities", "v", val=np.zeros((3 * (nx - 1) * (ny - 1))))#rows=[0], cols=[0], val=[0]
        self.declare_partials("freestream_velocities", "mesh", rows=[0], cols=[0], val=[0])
        self.declare_partials("freestream_velocities", "velocity_distribution", rows=rows, cols=cols)

        self.set_check_partial_options("*", method='fd')

        if rotational:
            nn = 3 * system_size
            row_col = np.arange(nn)
            val = np.ones((nn,))
            self.declare_partials("freestream_velocities", "rotational_velocities", rows=row_col, cols=row_col, val=val)

    def compute(self, inputs, outputs):
        # Rotate the freestream velocities based on the angle of attack and the sideslip angle.

        alpha = inputs["alpha"][0] * np.pi / 180.0
        beta = inputs["beta"][0] * np.pi / 180.0

        mesh = inputs["mesh"]
        nx, ny, _ = np.shape(mesh)
        vjet = inputs["velocity_distribution"]

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        # --- Here we inlcude the jet velocity ---
        v_tot = np.zeros(((nx-1)*(ny-1)))
        for i in range(nx-1):
            v_tot[i*(ny-1):(i+1)*(ny-1)] = vjet

        v_tot = v_tot.reshape((np.size(v_tot), 1))
        v_inf = v_tot * np.array([cosa * cosb, -sinb, sina * cosb])
        
        outputs["freestream_velocities"][:, :] = v_inf

        if self.options["rotational"]:
            outputs["freestream_velocities"][:, :] += inputs["rotational_velocities"]

    def compute_partials(self, inputs, J):
        alpha = inputs["alpha"][0] * np.pi / 180.0
        beta = inputs["beta"][0] * np.pi / 180.0
        v_adjusted = inputs["velocity_distribution"]
        mesh = inputs["mesh"]
        
        nx, ny, _ = np.shape(mesh)
        y = mesh[0, :, 1]
        y_ = np.zeros((len(y)-1))
        
        for i in range(len(y)-1):
            y_[i] = (y[i]+y[i+1])/2

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        # --- Here we inlcude the jet velocity ---
        v_tot = np.zeros(((nx-1)*(ny-1)))
        
        for i in range(nx-1):
            v_tot[i*(ny-1):(i+1)*(ny-1)] = v_adjusted[:]

        v_tot = v_tot.reshape((np.size(v_tot), 1))
        
        v_inf = np.ones(np.shape(v_tot)) * np.array([cosa * cosb, -sinb, sina * cosb])
        v_inf_a = v_tot[:] * np.array([-sina * cosb, 0.0, cosa * cosb]) * np.pi / 180.0
        v_inf_b = v_tot[:] * np.array([-cosa * sinb, -cosb, -sina * sinb]) * np.pi / 180.0

        J["freestream_velocities", "velocity_distribution"] = v_inf.flatten()

        J["freestream_velocities", "alpha"] = v_inf_a.flatten()

        J["freestream_velocities", "beta"] = v_inf_b.flatten()
