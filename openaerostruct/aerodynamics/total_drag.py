import openmdao.api as om


class TotalDrag(om.ExplicitComponent):
    """Calculate total drag in force units.

    parameters
    ----------
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.
    CDv : float
        Calculated coefficient of viscous drag for the lifting surface.

    Returns
    -------
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        surface = self.options["surface"]

        self.add_input("CDi", val=1.0)
        self.add_input("CDv", val=1.0)
        self.add_input("CDw", val=1.0)
        
        self.add_input("S_ref", val=1.0, units="m**2")
        self.add_input("v", val=1.0, units="m/s")
        self.add_input("rho", val=1.0, units="kg/m**3")

        self.add_output("CD", val=1.0, tags=["mphys_result"])
        self.add_output("D_total", val=1.0)

        self.CD0 = surface["CD0"]

        self.declare_partials("CD", "CDi", val=1.0)
        self.declare_partials("CD", "CDv", val=1.0)
        self.declare_partials("CD", "CDw", val=1.0)
        self.declare_partials("D_total", "CDi")
        self.declare_partials("D_total", "CDv")
        self.declare_partials("D_total", "CDw")
        self.declare_partials("D_total", "S_ref")
        self.declare_partials("D_total", "v")
        self.declare_partials("D_total", "rho")

    def compute(self, inputs, outputs):
        S_ref = inputs["S_ref"]
        rho = inputs["rho"]
        v = inputs["v"]
        
        outputs["CD"] = inputs["CDi"] + inputs["CDv"] + inputs["CDw"] + self.CD0
        outputs["D_total"] = outputs["CD"] * (0.5 * rho * v**2 * S_ref)
        
    def compute_partials(self, inputs, partials):
        S_ref = inputs["S_ref"]
        rho = inputs["rho"]
        v = inputs["v"]
        CDi = inputs["CDi"]
        CDv = inputs["CDv"]
        CDw = inputs["CDw"]
        
        CDtotal = CDi + CDv + CDw + self.CD0
        
        partials["D_total", "CDi"] = (0.5 * rho * v**2 * S_ref)
        partials["D_total", "CDv"] = (0.5 * rho * v**2 * S_ref)
        partials["D_total", "CDw"] = (0.5 * rho * v**2 * S_ref)
        
        partials["D_total", "S_ref"] = CDtotal * (0.5 * rho * v**2)
        partials["D_total", "rho"] = CDtotal * (0.5 * v**2 * S_ref)
        partials["D_total", "v"] = CDtotal * (rho * v * S_ref)