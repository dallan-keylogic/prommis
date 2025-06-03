#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Solvent Extraction Model

========================

Author: Arkoprabho Dasgupta

The Solvent Extraction unit model is used to perform the solvent extraction unit operation.
It represents a series of tanks, referred to as stages, through which the aqueous and organic
phases are passed, and the desired components are extracted subsequently.

Configuration Arguments
-----------------------

The user must specify the following configurations in a solvent extraction model to be able to
use it.

The user must specify the aqueous feed input in the ``aqueous_stream`` configuration, with a
configuration that describes the aqueous feed's properties.

The user must specify the organic feed input in the ``organic_stream`` configuration, with a
configuration that describes the organic feed's properties.

The number of stages in the solvent extraction process has to be specified by the user through
the ``number_of_finite_elements`` configuration. It takes an integer value.


Stream configurations
---------------------

Each of the feed streams has to have a dictionary that specifies the property packages and other
details as mentioned below.

The ``property_package`` configuration is the property package that describes the state conditions
and properties of a particular stream.

The ``property_package_args`` configuration is any specific set of arguments that has to be passed
to the property block for the unit operation.

The user can specify the direction of the flow of the stream through the stages through the
configuration ``flow_direction``. This is a configuration, that uses FlowDirection Enum, which
can have two possible values.

Degrees of freedom
------------------

When the solvent extraction model is operated in steady state, the number of degrees of freedom of
the model is equal to the sum of the number of distribution coefficients of the total components
involved in the mass transfer operation and the volumes and volume fractions, for all the stages.

If the model is operated in dynamic state, the number of degrees of freedom is equal to the sum
of the distribution coefficient of all components involved in the mass transfer operation, values
of the state block variables of all the components of the system at the start of the operation, the
volumes and the volume fractions, for all the stages.

Model structure
---------------

The core model consists of a MSContactor model, with stream names hard coded as 'aqueous' and
'organic', and the stream dictionaries and number of finite elements are the same as those provided
by the user.

This model uses the heterogeneous reaction term defined in the MSContactor, to calculate the amount of
material transferred between the phases for each of the rare earth elements. The distribution coefficients
used for this quantification are defined in the reaction package, and the constraint pertaining to the
distribution coefficient is defined in the solvent extraction model.

The pressure buildup in each of the stages has been defined in the model. For defining the pressure, we
need the volume of the phases, so the configuration ``has_holdup`` has to be set to True to obtain the
pressure of the phases.

"""

from pyomo.common.config import Bool, ConfigDict, ConfigValue, In
from pyomo.environ import Block,Constraint, Param, Reference, Suffix, TransformationFactory, units, Var
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.opt.results.solver import check_optimal_termination
from pyomo.network import Port
from pyomo.dae.flatten import slice_component_along_sets, flatten_dae_components

from idaes.core import (
    FlowDirection,
    UnitModelBlockData,
    declare_process_block_class,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.constants import Constants
from idaes.core.initialization import ModularInitializerBase, BlockTriangularizationInitializer
from idaes.core.scaling import ConstraintScalingScheme, CustomScalerBase, get_scaling_factor
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.models.unit_models.mscontactor import MSContactor

class SolventExtractionScaler(CustomScalerBase):
    """
    Scaler for the SolventExtraction unit model.
    """

    def variable_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Variable scaling routine for SolventExtraction.

        Args:
            model: instance of SolventExraction to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: dict of Scalers to use for sub-models, keyed by submodel local name

        Returns:
            None
        """
        if submodel_scalers is None:
            submodel_scalers = {}

        # There are no Vars besides those created by the MSContactor
        self.call_submodel_scaler_method(
            submodel=model.mscontactor,
            submodel_scalers=submodel_scalers,
            method="variable_scaling_routine",
            overwrite=overwrite,
        )

                    
    def constraint_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Routine to apply scaling factors to constraints in model.

        Args:
            model: model to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: dict of Scalers to use for sub-models, keyed by submodel local name

        Returns:
            None
        """

        self.call_submodel_scaler_method(
            submodel=model.mscontactor,
            submodel_scalers=submodel_scalers,
            method="constraint_scaling_routine",
            overwrite=overwrite,
        )
        for idx, con in model.distribution_extent_constraint.items():
            t, e, j_aq, j_o = idx
            self.scale_constraint_by_variable(
                con,
                model.mscontactor.organic[t, e].conc_mol_comp[j_o],
                overwrite=overwrite
            )

        for idx, con in model.aqueous_pressure_constraint.items():
            t, e = idx
            self.scale_constraint_by_variable(
                con,
                model.mscontactor.aqueous[t, e].pressure,
                overwrite=overwrite
            )
        for idx, con in model.organic_pressure_constraint.items():
            t, e = idx
            self.scale_constraint_by_variable(
                con,
                model.mscontactor.organic[t, e].pressure,
                overwrite=overwrite
            )

class SolventExtractionInitializer(ModularInitializerBase):
    """
    This is a general purpose Initializer  for the Solvent Extraction unit model.

    This routine calls the initializer for the internal MSContactor model.

    """

    CONFIG = ModularInitializerBase.CONFIG()

    CONFIG.declare(
        "ssc_solver_options",
        ConfigDict(
            implicit=True,
            description="Dict of arguments for solver calls by ssc_solver",
        ),
    )
    CONFIG.declare(
        "calculate_variable_options",
        ConfigDict(
            implicit=True,
            description="Dict of options to pass to 1x1 block solver",
            doc="Dict of options to pass to calc_var_kwds argument in "
            "scc_solver method.",
        ),
    )

    def initialize_main_model(
        self,
        model: Block,
    ):
        """
        Initialization routine for MSContactor Blocks.

        Args:
            model: model to be initialized

        Returns:
            None
        """

        model.mscontactor.heterogeneous_reaction_extent.fix(1e-8)
        model.mscontactor.volume.fix(1)
        model.mscontactor.volume_frac_stream[:, :, "aqueous"].fix(0.5)

        # Initialize MSContactor
        msc_init = model.mscontactor.default_initializer(
            ssc_solver_options=self.config.ssc_solver_options,
            calculate_variable_options=self.config.calculate_variable_options,
        )
        msc_init.initialize(model.mscontactor)

        bt_init = BlockTriangularizationInitializer(
            block_solver_options=self.config.ssc_solver_options,
            block_solver_call_options={
                "tee": True
            },
            calculate_variable_options=self.config.calculate_variable_options
        )
        # import pdb; pdb.set_trace()
        model.mscontactor.heterogeneous_reaction_extent.unfix()

        other_vars, element_vars = flatten_dae_components(model, model.mscontactor.elements, ctype=Var)
        other_cons, element_cons = flatten_dae_components(model, model.mscontactor.elements, ctype=Constraint)

        solver = self._get_solver()
        # TransformationFactory("contrib.strip_var_bounds").apply_to(model, reversible=True)
        # import pdb; pdb.set_trace()
        # for e in model.mscontactor.elements:
        #     self.restore_model_state(model)
        #     self.fix_initialization_states(model)

        #     for var in other_vars:
        #         var.fix()
        #     for con in other_cons:
        #         con.deactivate()

        #     for e2 in model.mscontactor.elements:
        #         if e2 != e:
        #             for var in element_vars:
        #                 var[e2].fix()
        #             for con in element_cons:
        #                 con[e2].deactivate()
        #     assert degrees_of_freedom(model) == 0
        #     print(f"Initializing element {e}")
        #     # model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        #     # model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        #     for idx in model.mscontactor.heterogeneous_reactions[0, e].distribution_coefficient:
        #         calculate_variable_from_constraint(
        #             model.mscontactor.heterogeneous_reactions[0, e].distribution_coefficient[idx],
        #             model.mscontactor.heterogeneous_reactions[0, e].distribution_constraint[idx],
        #             # **self.config.calculate_variable_options
        #         )
        #     model.mscontactor.heterogeneous_reactions[0, e].distribution_constraint.deactivate()
        #     model.mscontactor.heterogeneous_reactions[0, e].distribution_coefficient.fix()
        #     bt_init.initialize(model)
        #     model.mscontactor.heterogeneous_reactions[0, e].distribution_constraint.activate()
        #     model.mscontactor.heterogeneous_reactions[0, e].distribution_coefficient.unfix()
        #     results = solver.solve(model, tee=True)
        #     if not check_optimal_termination(results):
        #         from idaes.core.util.model_diagnostics import DiagnosticsToolbox
        #         diag_tbx = DiagnosticsToolbox(model)
        #         import pdb; pdb.set_trace()

        # self.restore_model_state(model)
        # self.fix_initialization_states(model)
        import pdb; pdb.set_trace()
        init_model = solver.solve(model, tee=True)
        # import pdb; pdb.set_trace()
        # TransformationFactory("contrib.strip_var_bounds").revert(model)

        return init_model


Stream_Config = ConfigDict()

Stream_Config.declare(
    "property_package",
    ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for given stream",
        doc="""Property parameter object used to define property calculations for given stream,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
    ),
)

Stream_Config.declare(
    "property_package_args",
    ConfigDict(
        implicit=True,
        description="Dict of arguments to use for constructing property package",
        doc="""A ConfigDict with arguments to be passed to property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
    ),
)

Stream_Config.declare(
    "flow_direction",
    ConfigValue(
        default=FlowDirection.forward,
        domain=In(FlowDirection),
        doc="Direction of flow for stream",
        description="FlowDirection Enum indicating direction of "
        "flow for given stream. Default=FlowDirection.forward.",
    ),
)

Stream_Config.declare(
    "has_energy_balance",
    ConfigValue(
        default=False,
        domain=Bool,
        doc="Bool indicating whether to include energy balance for stream. Default=False.",
    ),
)

Stream_Config.declare(
    "has_pressure_balance",
    ConfigValue(
        default=False,
        domain=In([False]),
        doc="Bool indicating whether to include pressure balance for stream. Default=False.",
    ),
)


@declare_process_block_class("SolventExtraction")
class SolventExtractionData(UnitModelBlockData):

    default_initializer = SolventExtractionInitializer
    default_scaler = SolventExtractionScaler

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "aqueous_stream",
        Stream_Config(
            description="Aqueous stream properties",
        ),
    )

    CONFIG.declare(
        "organic_stream",
        Stream_Config(
            description="Organic stream properties",
        ),
    )

    CONFIG.declare(
        "number_of_finite_elements",
        ConfigValue(domain=int, description="Number of finite elements to use"),
    )

    CONFIG.declare(
        "reaction_package",
        ConfigValue(
            # TODO: Add a domain validator for this
            description="Heterogeneous reaction package for leaching.",
        ),
    )
    CONFIG.declare(
        "reaction_package_args",
        ConfigValue(
            default=None,
            domain=dict,
            description="Arguments for heterogeneous reaction package for leaching.",
        ),
    )

    def build(self):
        super().build()

        streams_dict = {
            "aqueous": self.config.aqueous_stream,
            "organic": self.config.organic_stream,
        }
        self.mscontactor = MSContactor(
            streams=streams_dict,
            number_of_finite_elements=self.config.number_of_finite_elements,
            heterogeneous_reactions=self.config.reaction_package,
            heterogeneous_reactions_args=self.config.reaction_package_args,
            has_holdup=self.config.has_holdup,
        )

        distribution_set = [
            ("Al", "Al_o"),
            ("Ca", "Ca_o"),
            ("Fe", "Fe_o"),
            ("Sc", "Sc_o"),
            ("Y", "Y_o"),
            ("Gd", "Gd_o"),
            ("Dy", "Dy_o"),
            ("Sm", "Sm_o"),
            ("La", "La_o"),
            ("Pr", "Pr_o"),
            ("Ce", "Ce_o"),
            ("Nd", "Nd_o"),
        ]

        def distribution_ratio_rule(b, t, s, e, f):
            return (
                b.mscontactor.organic[t, s].conc_mol_comp[f]
                == b.mscontactor.heterogeneous_reactions[t, s].distribution_coefficient[
                    e
                ]
                * b.mscontactor.aqueous[t, s].conc_mol_comp[e]
            )

        self.distribution_extent_constraint = Constraint(
            self.flowsheet().time,
            self.mscontactor.elements,
            distribution_set,
            rule=distribution_ratio_rule,
        )

        self.area_cross_stage = Param(
            self.mscontactor.elements,
            units=units.m**2,
            doc="Cross sectional area stage",
            initialize=1,
            mutable=True,
        )

        self.elevation = Param(
            self.mscontactor.elements,
            units=units.m,
            doc="Elevation of each stage",
            initialize=1,
            mutable=True,
        )

        def aqueous_pressure_calculation(b, t, s):
            # g = 9.8 * (units.m) / units.sec**2
            g = Constants.acceleration_gravity
            P_atm = 101325 * units.Pa

            rho_aq = sum(
                b.mscontactor.aqueous[t, s].conc_mass_comp[p]
                for p in getattr(b.mscontactor, "aqueous").component_list
            )
            rho_og = sum(
                b.mscontactor.organic[t, s].conc_mass_comp[p]
                for p in getattr(b.mscontactor, "organic").component_list
            )
            P_aq = units.convert(
                (
                    rho_aq
                    * g
                    * (
                        b.mscontactor.volume[s]
                        * b.mscontactor.volume_frac_stream[t, s, "aqueous"]
                        / b.area_cross_stage[s]
                        + b.elevation[s]
                    )
                ),
                to_units=units.Pa,
            )
            P_org = units.convert(
                (
                    rho_og
                    * g
                    * b.mscontactor.volume[s]
                    * b.mscontactor.volume_frac_stream[t, s, "organic"]
                    / b.area_cross_stage[s]
                ),
                to_units=units.Pa,
            )
            return b.mscontactor.aqueous[t, s].pressure == P_aq + P_org + P_atm

        self.aqueous_pressure_constraint = Constraint(
            self.flowsheet().time,
            self.mscontactor.elements,
            rule=aqueous_pressure_calculation,
        )

        def organic_pressure_calculation(b, t, s):
            g = 9.8 * (units.m) / units.sec**2
            P_atm = 101325 * units.Pa

            rho_og = sum(
                b.mscontactor.organic[t, s].conc_mass_comp[p]
                for p in getattr(b.mscontactor, "organic").component_list
            )

            P_org = units.convert(
                (
                    rho_og
                    * g
                    * b.mscontactor.volume[s]
                    * b.mscontactor.volume_frac_stream[t, s, "organic"]
                    / b.area_cross_stage[s]
                ),
                to_units=units.Pa,
            )
            return b.mscontactor.organic[t, s].pressure == P_org + P_atm

        self.organic_pressure_constraint = Constraint(
            self.flowsheet().time,
            self.mscontactor.elements,
            rule=organic_pressure_calculation,
        )

        self.aqueous_inlet = Port(extends=self.mscontactor.aqueous_inlet)
        self.aqueous_outlet = Port(extends=self.mscontactor.aqueous_outlet)
        self.organic_inlet = Port(extends=self.mscontactor.organic_inlet)
        self.organic_outlet = Port(extends=self.mscontactor.organic_outlet)