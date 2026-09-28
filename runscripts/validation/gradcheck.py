"""Run `torch.autograd.gradcheck` on the PISO and MHD solver adjoints.

Usage:
    python runscripts/validation/gradcheck.py --suite all --advection-scheme central
"""

import torch

import fluidgym.simulation.pict.data.shapes as shapes
from fluidgym.simulation.extensions import PISOtorch  # type: ignore
from fluidgym.simulation.mhd_simulation import _compute_u_cross_eb_flat
from fluidgym.simulation.pict import PISOtorch_diff
from fluidgym.simulation.simulation import Simulation

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

import logging

LOG = logging.getLogger("gradcheck")

DTYPE = torch.float64
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-5
GRADCHECK_NONDET_TOL = 1e-7

USE_RANDOM_INPUTS = True

USE_SCALAR_VISCOSITY = False
USE_BLOCK_VISCOSITY = False

# ----------------------------------------------------------------------------
# Convective scheme (see schemes.md)
# ----------------------------------------------------------------------------
# Every domain built by a setup function gets ADVECTION_SCHEME applied before the
# test runs, so the whole suite can be re-run per scheme.
ADVECTION_SCHEMES = {
    "central":       PISOtorch.AdvectionScheme.CENTRAL,
    "linear_upwind": PISOtorch.AdvectionScheme.LINEAR_UPWIND,
}

ADVECTION_SCHEME = "central"

# Seed used when two domains must be built with identical random fields.
COMPARE_SEED = 0


def _apply_advection_scheme(setup_result):
    """Apply ADVECTION_SCHEME to the domain returned by a setup function.

    Setup functions return either a bare Domain or a tuple with the Domain first
    (PISO: (domain, is_non_ortho, prep_fn); MHD: (domain, e_b)).
    """
    domain = setup_result[0] if isinstance(setup_result, tuple) else setup_result
    domain.setAdvectionScheme(ADVECTION_SCHEMES[ADVECTION_SCHEME])
    domain.UpdateDomainData()
    return setup_result


def _with_advection_scheme(setup_fn):
    """Wrap a setup function so the current ADVECTION_SCHEME is applied to it."""
    def wrapped(*args, **kwargs):
        return _apply_advection_scheme(setup_fn(*args, **kwargs))
    return wrapped


# these must match the definition in 'PISO_multiblock_cuda.h'
NON_ORTHO_DIRECT_MATRIX = 1
NON_ORTHO_DIRECT_RHS = 2 # less stable than NON_ORTHO_DIRECT_MATRIX
NON_ORTHO_DIAGONAL_MATRIX = 4 # not implemented
NON_ORTHO_DIAGONAL_RHS = 8
NON_ORTHO_CENTER_MATRIX = 16

NON_ORTHO_FLAGS = NON_ORTHO_CENTER_MATRIX | NON_ORTHO_DIRECT_MATRIX | NON_ORTHO_DIAGONAL_RHS # Bit flags
PRESSURE_FACE_TRANSFORM = False
PRESSURE_TIME_STEP_NORM = False
VELOCITY_CORRECTOR_VERSION = 1 # finite differencing

def gradcheck_LinearSolve(mat, mat_value, rhs, use_BiCG):
    x = PISOtorch_diff.linear_solve_GPU(mat, rhs, False, use_BiCG, return_best_result=not use_BiCG)
    #if not use_BiCG:
    #    x = x - torch.mean(x)
    return x

def test_LinearSolve_Scalar(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()

    if domain.getPassiveScalarChannels() == 0:
        LOG.warning("gradcheck LinearSolve_Scalar: skipped (domain has no passive scalar channels)")
        return

    has_scalar = domain.getPassiveScalarChannels() > 0
    sim = Simulation(domain=domain, dt=time_step.item(), prep_fn=prep_fn, non_orthogonal=is_non_ortho,
        pressure_non_ortho_steps=2 if is_non_ortho else 1,
        advect_passive_scalar=has_scalar)
    sim.make_divergence_free()

    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS, forPassiveScalar=True, passiveScalarChannel=0)
    PISOtorch_diff.SetupAdvectionScalar(domain, time_step, NON_ORTHO_FLAGS)
    domain.C.value.requires_grad_(True)
    domain.scalarRHS.requires_grad_(True)

    inputs = (domain.C, domain.C.value, domain.scalarRHS, True)
    test = torch.autograd.gradcheck(gradcheck_LinearSolve, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    LOG.info("gradcheck LinearSolve_Scalar: %s", test)

def test_LinearSolve_Velocity(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()

    has_scalar = domain.getPassiveScalarChannels() > 0
    sim = Simulation(domain=domain, dt=time_step.item(), prep_fn=prep_fn, non_orthogonal=is_non_ortho,
        pressure_return_best_result=True,
        pressure_non_ortho_steps=2 if is_non_ortho else 1,
        advect_passive_scalar=has_scalar)
    sim.make_divergence_free()

    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
    PISOtorch_diff.SetupAdvectionVelocity(domain, time_step, NON_ORTHO_FLAGS)
    domain.C.value.requires_grad_(True)
    domain.velocityRHS.requires_grad_(True)

    inputs = (domain.C, domain.C.value, domain.velocityRHS, True)
    test = torch.autograd.gradcheck(gradcheck_LinearSolve, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    LOG.info("gradcheck LinearSolve_Velocity: %s", test)

def test_LinearSolve_Pressure(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()

    has_scalar = domain.getPassiveScalarChannels() > 0
    sim = Simulation(domain=domain, dt=time_step.item(), prep_fn=prep_fn, non_orthogonal=is_non_ortho,
        pressure_return_best_result=True,
        pressure_non_ortho_steps=2 if is_non_ortho else 1,
        advect_passive_scalar=has_scalar)
    sim.make_divergence_free(4)

    # Build a realistic pressure system: run the prediction (advection) step
    # manually so that A and velocityResult are consistent, then call
    # SetupPressureCorrection — mirrors the old StopHandler approach.
    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
    PISOtorch_diff.SetupAdvectionVelocity(domain, time_step, NON_ORTHO_FLAGS)
    vel_result = PISOtorch_diff.linear_solve_GPU(domain.C, domain.velocityRHS, False, True)
    domain.setVelocityResult(vel_result)
    PISOtorch.CopyVelocityResultToBlocks(domain)
    domain.UpdateDomainData()

    LOG.info("LinearSolve_Pressure: SetupPressureCorrection")
    PISOtorch_diff.SetupPressureCorrection(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)

    domain.pressureRHSdiv.requires_grad_(True)
    # SetupPressureCorrection rebinds P.value / pressureRHS / pressureRHSdiv, so
    # the domain's device-side pointers are stale until this runs. Without it the
    # solve aborts with "Domain's tensors have been changed".
    domain.UpdateDomainData()

    inputs = (domain.P, domain.P.value, domain.pressureRHSdiv, False)
    test = torch.autograd.gradcheck(gradcheck_LinearSolve, inputs, eps=GRADCHECK_EPS if GRADCHECK_EPS < 1e-8 else 1e-8,
        atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach()  # as in the other tests: frees the domain after autograd
    LOG.info("gradcheck LinearSolve_Pressure: %s", test)

def gradcheck_SetupAdvectionMatrix(domain, time_step, for_passive_scalar, *block_tensors):
    return PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS, for_passive_scalar)

def test_SetupAdvectionMatrix_base(domain_setup_fn, time_step, for_passive_scalar):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupAdvectionMatrix: using random inputs")
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("randomize boundary[%d] velocity", bound_idx)
                bound.setVelocity(torch.randn_like(bound.velocity))

    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionMatrix: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)

    use_scalar_viscosity = USE_SCALAR_VISCOSITY
    if use_scalar_viscosity:
        LOG.info("test_SetupAdvectionMatrix: using random scalar viscosity")
        scalar_viscosity = domain.viscosity.to(cuda_device)
        scalar_viscosity = scalar_viscosity + scalar_viscosity*0.5*torch.randn([domain.getPassiveScalarChannels()], dtype=DTYPE, device=cuda_device)
        domain.setScalarViscosity(scalar_viscosity)

    tensor_filter = ["VELOCITY", "BOUNDARY_VELOCITY", "VISCOSITY"]
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    if use_scalar_viscosity:
        tensor_filter.append("PASSIVE_SCALAR_VISCOSITY")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, for_passive_scalar, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupAdvectionMatrix, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupAdvectionMatrix: %s", test)

def test_SetupAdvectionMatrix(domain_setup_fn, time_step):
    test_SetupAdvectionMatrix_base(domain_setup_fn, time_step, False)

def test_SetupAdvectionMatrix_scalar(domain_setup_fn, time_step):
    test_SetupAdvectionMatrix_base(domain_setup_fn, time_step, True)

def gradcheck_SetupAdvectionScalar(domain, time_step, *block_tensors):
    #domain.setScalarResult(block_tensors[-1])
    #domain.UpdateDomainData()
    return PISOtorch_diff.SetupAdvectionScalar(domain, time_step, NON_ORTHO_FLAGS)

def test_SetupAdvectionScalar(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyScalarResultFromBlocks(domain)

    if USE_RANDOM_INPUTS:
        LOG.info("SetupAdvectionScalar: using random inputs")
        for block in domain.getBlocks():
            block.setPassiveScalar(torch.randn_like(block.passiveScalar))
            for bound_idx, bound in block.getFixedBoundaries():
                bound.setPassiveScalar(torch.randn_like(bound.passiveScalar))
                bound.setVelocity(torch.randn_like(bound.velocity))
        domain.setScalarResult(torch.randn_like(domain.scalarResult))

    use_scalar_viscosity = USE_SCALAR_VISCOSITY
    if use_scalar_viscosity:
        LOG.info("SetupAdvectionScalar: using random scalar viscosity")
        scalar_viscosity = domain.viscosity.to(cuda_device)
        scalar_viscosity = scalar_viscosity + scalar_viscosity*0.5*torch.randn([domain.getPassiveScalarChannels()], dtype=DTYPE, device=cuda_device)
        domain.setScalarViscosity(scalar_viscosity)

    tensor_filter = ["PASSIVE_SCALAR", "BOUNDARY_PASSIVE_SCALAR", "BOUNDARY_VELOCITY", "VISCOSITY"] #, "VISCOSITY"]
    if is_non_ortho:
        tensor_filter.append("PASSIVE_SCALAR_RESULT")
    if use_scalar_viscosity:
        tensor_filter.append("PASSIVE_SCALAR_VISCOSITY")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)



    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupAdvectionScalar, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupAdvectionScalar: %s", test)

def gradcheck_SetupAdvectionVelocity(domain, time_step, *block_tensors):
    return PISOtorch_diff.SetupAdvectionVelocity(domain, time_step, NON_ORTHO_FLAGS, False)

def test_SetupAdvectionVelocity(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyVelocityResultFromBlocks(domain)

    tensor_filter = ["VELOCITY", "BOUNDARY_VELOCITY", "VISCOSITY"] #, "VISCOSITY"]

    if True:
        LOG.info("test_SetupAdvectionVelocity: using varying boundaries")
        for block in domain.getBlocks():
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("varying boundary[%d] velocity", bound_idx)
                bound.makeVelocityVarying()

    if False:
        LOG.info("test_SetupAdvectionVelocity: using random velocity source")
        tensor_filter.append("VELOCITY_SOURCE")
        for block in domain.getBlocks():
            block.setVelocitySource(torch.randn_like(block.velocity))

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupAdvectionVelocity: using random inputs")
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("randomize boundary[%d] velocity", bound_idx)
                bound.setVelocity(torch.randn_like(bound.velocity))
        domain.setVelocityResult(torch.randn_like(domain.velocityResult))

    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionVelocity: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)

    if is_non_ortho:
        tensor_filter.append("VELOCITY_RESULT")
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupAdvectionVelocity, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupAdvectionVelocity: %s", test)


# ----------------------------------------------------------------------------
# Convective scheme tests
# ----------------------------------------------------------------------------

def _as_tensor_list(out) -> list:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    return []


def _advection_forward(domain_setup_fn, time_step, scheme_name: str):
    """Forward the advection operator under `scheme_name` from a seeded domain.

    Returns the implicit matrix (C.value and its diagonal A) and the explicit
    SetupAdvectionVelocity output.
    """
    torch.manual_seed(COMPARE_SEED)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    # Overrides whatever the runner's wrapper applied.
    domain.setAdvectionScheme(ADVECTION_SCHEMES[scheme_name])
    domain.UpdateDomainData()
    PISOtorch.CopyVelocityResultFromBlocks(domain)

    for block in domain.getBlocks():
        block.setVelocity(torch.randn_like(block.velocity))
    domain.setVelocityResult(torch.randn_like(domain.velocityResult))
    domain.UpdateDomainData()

    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
    tensors = [domain.C.value.detach().clone(), domain.A.detach().clone()]

    out = PISOtorch_diff.SetupAdvectionVelocity(domain, time_step, NON_ORTHO_FLAGS, False)
    tensors += [t.detach().clone() for t in _as_tensor_list(out)]
    domain.Detach()
    return tensors


def test_AdvectionSchemeActive(domain_setup_fn, time_step):
    """Assert the selected scheme actually changes the discretization."""
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)

    ref = _advection_forward(domain_setup_fn, time_step, "central")
    got = _advection_forward(domain_setup_fn, time_step, ADVECTION_SCHEME)

    assert len(ref) == len(got) and len(ref) > 0, (
        f"advection forward returned {len(ref)} vs {len(got)} tensors"
    )
    max_diff = max(
        (a - b).abs().max().item() for a, b in zip(ref, got)
    )

    if ADVECTION_SCHEME == "central":
        assert max_diff == 0.0, (
            f"'central' must reproduce itself exactly, got max|diff| = {max_diff:.3e}"
        )
        LOG.info("AdvectionSchemeActive: central is bit-identical, as expected")
    else:
        assert max_diff > 0.0, (
            f"scheme '{ADVECTION_SCHEME}' produced a matrix and RHS identical to "
            "'central'. No face had a far-upwind cell, so neither the implicit "
            "upwind branch nor the deferred correction ran, and any gradcheck of "
            "this domain is vacuous. Use a larger domain."
        )
        LOG.info(
            "AdvectionSchemeActive: %s differs from central by max|diff| = %.3e",
            ADVECTION_SCHEME, max_diff,
        )


def gradcheck_CopyScalarResultToBlocks(domain, *block_tensors):
    PISOtorch_diff.CopyScalarResultToBlocks(domain)
    return domain.scalarResult

def gradcheck_SetupPressureCorrection(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureCorrection(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.P.value, domain.pressureRHS, domain.pressureRHSdiv

def test_SetupPressureCorrection(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)
    tensor_filter = ["VELOCITY", "VELOCITY_RESULT", "BOUNDARY_VELOCITY", "VISCOSITY", "A", "C"] #"C"

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureCorrection: using random inputs")
        domain.C.setValue(torch.randn_like(domain.C.value))
        domain.setA(torch.randn_like(domain.A)) # should be the diagonal of C, but should not matter for the gradcheck
        domain.setVelocityResult(torch.randn_like(domain.velocityResult))
        #domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionVelocity: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)

    if is_non_ortho:
        tensor_filter.append("PRESSURE_RESULT")
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    _, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureCorrection, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureCorrection: %s", test)

def gradcheck_SetupPressureMatrix(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureMatrix(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM)
    return domain.P.value

def test_SetupPressureMatrix(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, _, _ = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)
    tensor_filter = ["A"]

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureMatrix: using random inputs")
        domain.setA(torch.randn_like(domain.A))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity()
            domain.getBlock(blockIdx).CreatePressure()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureMatrix, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureMatrix: %s", test)

def gradcheck_SetupPressureRHS(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureRHS(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.pressureRHS, domain.pressureRHSdiv

def test_SetupPressureRHS(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)
    tensor_filter = ["VELOCITY", "VELOCITY_RESULT", "BOUNDARY_VELOCITY", "VISCOSITY", "C", "A"] #

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureRHS: using random inputs")
        domain.C.setValue(torch.randn_like(domain.C.value))
        domain.setA(torch.randn_like(domain.A)) # should be the diagonal of C, but should not matter for the gradcheck
        domain.setVelocityResult(torch.randn_like(domain.velocityResult))
        #domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionVelocity: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)

    if is_non_ortho:
        tensor_filter.append("PRESSURE_RESULT")
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    _, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors:
        if t is not None:
            t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureRHS, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureRHS: %s", test)

def gradcheck_SetupPressureRHSdiv(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureRHSdiv(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.pressureRHSdiv

def test_SetupPressureRHSdiv(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureRHSdiv: using random inputs")
        domain.setA(torch.randn_like(domain.A))
        domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

        domain.setPressureRHS(domain.velocityResult.clone())
        domain.UpdateDomainData()

    tensor_filter = ["PRESSURE_RHS", "BOUNDARY_VELOCITY", "A"] #
    if is_non_ortho:
        tensor_filter.append("PRESSURE_RESULT")
        tensor_filter.append("A")
    _, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureRHSdiv, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureRHSdiv: %s", test)

def gradcheck_CorrectVelocity(domain, time_step, *tensors):
    PISOtorch_diff.CorrectVelocity(domain, time_step, VELOCITY_CORRECTOR_VERSION, PRESSURE_TIME_STEP_NORM)
    return domain.velocityResult

def test_CorrectVelocity(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()



    if USE_RANDOM_INPUTS:
        LOG.info("test_CorrectVelocity: using random inputs")
        domain.setA(torch.randn_like(domain.A)) # should be the diagonal of C, but should not matter for the gradcheck
        domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        for block in domain.getBlocks():
        #    block.setVelocity(torch.randn_like(block.velocity))
            block.setPressure(torch.randn_like(block.pressure))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

        PISOtorch_diff.SetupPressureCorrection(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
        pressureResult = PISOtorch_diff.linear_solve_GPU(domain.P, domain.pressureRHSdiv, use_BiCG=False)
        domain.setPressureResult(pressureResult)
        domain.UpdateDomainData()
        PISOtorch_diff.CopyPressureResultToBlocks(domain)

    tensor_filter = ["PRESSURE", "PRESSURE_RHS", "A"] #
    _, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)

    test = torch.autograd.gradcheck(gradcheck_CorrectVelocity, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck CorrectVelocity: %s", test)


## FUSED TESTS

def gradcheck_dRHS_dA_dvisc(domain, time_step, *tensors):
    for_passive_scalar = False
    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS, for_passive_scalar)
    PISOtorch_diff.SetupPressureRHS(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.pressureRHS, domain.pressureRHSdiv

def test_dRHS_dA_dvisc(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)

    if USE_RANDOM_INPUTS:
        LOG.info("test_dRHS_dA_dvisc: using random inputs")
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("randomize boundary[%d] velocity", bound_idx)
                bound.setVelocity(torch.randn_like(bound.velocity))

        domain.setVelocityResult(torch.randn_like(domain.velocityResult))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
    else:
        raise NotImplementedError

    tensor_filter = ["VISCOSITY"]
    _, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors:
        if t is not None:
            t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_dRHS_dA_dvisc, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)

    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck dRHS_dA_dvisc: %s", test)


def gradcheck_advect_static(domain, time_step, *block_tensors):
    raise NotImplementedError

def gradcheck_div_free(domain, time_step, *velocities):
    raise NotImplementedError


def get_test_params(domain_setup_fn):
    return {"domain_setup_fn": domain_setup_fn, "time_step": 0.5}


# ============================================================================
# PISO DOMAIN SETUPS (adapted from test_setups.py)
# ============================================================================

def _make_1block_piso(
    x: int, y: int, z: int = 0,
    vel=(1.0, 0.0),
    closed_bounds: bool = False,
    inflow_outflow: bool = False,
    viscosity: float = 0.0,
    domain_scale=None,
    transform_strength=None,
    rot_distortion_max_angle=None,
    vel_blob: bool = True,
):
    """Single-block PISO domain. Returns (domain, is_non_ortho, prep_fn).
    Equivalent to test_setups.make1BlockSetup with dtype=DTYPE.
    """
    if z > 0:
        dims = 3
        res = [z, y, x]
    else:
        dims = 2
        z = 0
        res = [y, x]

    is_non_ortho = False
    prep_fn: dict = {}

    viscosity_t = torch.tensor([viscosity], dtype=DTYPE, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity_t, "Domain1Block", dtype=DTYPE, device=cuda_device)

    data = shapes.get_grid_normal_dist(res, [0] * dims, [0.5] * dims)
    data = torch.reshape(data, [1, 1] + res).to(DTYPE).to(cuda_device).contiguous()

    velocity = (
        torch.tensor(list(vel), dtype=DTYPE, device=cuda_device)
        .reshape([1, dims] + [1] * dims)
        .repeat(1, 1, *res)
        .contiguous()
    )
    if vel_blob:
        velocity = velocity * data

    grid = None
    if domain_scale is not None or transform_strength is not None or rot_distortion_max_angle is not None:
        if domain_scale is None:
            domain_scale = [1] * dims
        if transform_strength is None:
            transform_strength = [1] * dims

        corner_upper = (x * domain_scale[0], y * domain_scale[1])
        grid = shapes.make_wall_refined_ortho_grid(
            x, y,
            corner_upper=corner_upper,
            wall_refinement=["-x", "+x", "-y", "+y"],
            base=transform_strength[:2],
            dtype=DTYPE,
        )

        if rot_distortion_max_angle is not None:
            max_r = min(corner_upper) * 0.5 * 0.9
            dist_scale = shapes.make_rotation_distance_scaling_fn_sine_half(
                rot_distortion_max_angle, max_r
            )
            grid = shapes.rotate_grid(grid, angle=rot_distortion_max_angle, distance_scaling=dist_scale)
            is_non_ortho = True

        if dims == 3:
            grid = shapes.extrude_grid_z(
                grid, z, end_z=z * domain_scale[2],
                weights_z="EXP", exp_base=transform_strength[2],
            )

        grid = grid.to(cuda_device).contiguous()

    block = domain.CreateBlock(velocity=velocity, passiveScalar=data, vertexCoordinates=grid, name="Block")

    if closed_bounds:
        block.CloseAllBoundaries()
    elif inflow_outflow:
        # Duct: Dirichlet inflow at -x, varying outflow at +x, walls elsewhere.
        block.CloseAllBoundaries()
        face = ([z] if dims == 3 else []) + [y, 1]
        bound_vel = torch.zeros([1, dims] + face, dtype=DTYPE, device=cuda_device)
        bound_vel[0, 0] = _BV
        block.getBoundary("-x").setVelocity(bound_vel)
        block.getBoundary("+x").setVelocity(bound_vel.clone())
        block.getBoundary("+x").makeVelocityVarying()

    domain.PrepareSolve()
    return domain, is_non_ortho, prep_fn


def _make_2block_piso_2d(
    x1: int, y: int, x2: int,
    vel1=(1.0, 0.0), vel2=(0.0, 1.0),
    vel_blob1: bool = True, vel_blob2: bool = True,
    closed_bounds: bool = False,
    viscosity: float = 0.0,
):
    """Two-block 2D PISO domain. Returns (domain, is_non_ortho, prep_fn).
    Equivalent to test_setups.make2BlockSetup2D with dtype=DTYPE.
    """
    dims = 2
    res1 = [y, x1]
    res2 = [y, x2]
    is_non_ortho = False
    prep_fn: dict = {}

    viscosity_t = torch.tensor([viscosity], dtype=DTYPE, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity_t, name="Domain2Block2D", dtype=DTYPE, device=cuda_device)

    data = shapes.get_grid_normal_dist(res1, [0] * dims, [0.5] * dims)
    data = torch.reshape(data, [1, 1] + res1).to(DTYPE).to(cuda_device)
    velocity = (
        torch.tensor(list(vel1), dtype=DTYPE, device=cuda_device)
        .reshape([1, dims] + [1] * dims)
        .repeat(1, 1, *res1)
        .contiguous()
    )
    if vel_blob1:
        velocity = velocity * data
    domain.CreateBlock(velocity=velocity, passiveScalar=data, name="Block 1")

    velocity2 = (
        torch.tensor(list(vel2), dtype=DTYPE, device=cuda_device)
        .reshape([1, dims] + [1] * dims)
        .repeat(1, 1, *res2)
        .contiguous()
    )
    if vel_blob2:
        velocity2 = velocity2 * 0
    domain.CreateBlock(velocity=velocity2, passiveScalar=None, name="Block 2")

    block, block2 = domain.getBlocks()
    if closed_bounds:
        block.CloseAllBoundaries()
        block2.CloseAllBoundaries()
    else:
        block.ConnectBlock("-x", block2, "+x", "-y")
    block.ConnectBlock("+x", block2, "-x", "-y")

    domain.PrepareSolve()
    return domain, is_non_ortho, prep_fn


_R  = 4    # BASE_RES
_BV = 0.8  # BASE_VEL
_VI = 0.3  # VISCOSITY
_S2  = [1/_R, 1/_R]                  # S_UNIFORM_2D
_SN2 = [1/_R, 1/(_R+1)]             # S_NORMALISED_2D  (RES_2D = [R, R+1])
_S3  = [1/_R, 1/_R, 1/_R]           # S_UNIFORM_3D

PISO_DOMAIN_SETUPS: dict = {
    # ---- 2D simple ----
    "1Block2D_velX+const_periodic":       lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), vel_blob=False),
    "1Block2D_velX+blob_periodic":        lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), vel_blob=True),
    "1Block2D_velX-blob_periodic":        lambda: _make_1block_piso(_R, _R+1, vel=(-_BV,0), vel_blob=True),
    "1Block2D_velY+blob_periodic":        lambda: _make_1block_piso(_R, _R+1, vel=(0,_BV), vel_blob=True),
    "1Block2D_velY-blob_periodic":        lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), vel_blob=True),
    "1Block2D_velX+blob_closed":          lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), closed_bounds=True, vel_blob=True),
    "2Block2D_velX+const_periodic":       lambda: _make_2block_piso_2d(_R, _R, _R, vel1=(_BV,0), vel2=(_BV,0), vel_blob1=False, vel_blob2=False),
    "2Block2D_velX+constY+_periodic":     lambda: _make_2block_piso_2d(_R, _R, _R, vel_blob1=False, vel_blob2=False),
    "2Block2D_velX+-blob_periodic":       lambda: _make_2block_piso_2d(_R, _R, _R, vel2=(-_BV,0), vel_blob1=True, vel_blob2=True),
    "2Block2D_velX+constY+_closed":       lambda: _make_2block_piso_2d(_R, _R, _R, closed_bounds=True, vel_blob1=False, vel_blob2=False),
    # ---- 2D viscosity ----
    "1Block2D_v_velX+const_periodic":     lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, vel_blob=False),
    "1Block2D_v_velX+blob_periodic":      lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, vel_blob=True),
    "1Block2D_v_velX-blob_periodic":      lambda: _make_1block_piso(_R, _R+1, vel=(-_BV,0), viscosity=_VI, vel_blob=True),
    "1Block2D_v_velY+blob_periodic":      lambda: _make_1block_piso(_R, _R+1, vel=(0,_BV), viscosity=_VI, vel_blob=True),
    "1Block2D_v_velY-blob_periodic":      lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), viscosity=_VI, vel_blob=True),
    "1Block2D_v_velX+blob_closed":        lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, closed_bounds=True, vel_blob=True),
    "2Block2D_v_velX+const_periodic":     lambda: _make_2block_piso_2d(_R, _R, _R, vel1=(_BV,0), vel2=(_BV,0), viscosity=_VI, vel_blob1=False, vel_blob2=False),
    "2Block2D_v_velX+constY+_periodic":   lambda: _make_2block_piso_2d(_R, _R, _R, viscosity=_VI, vel_blob1=False, vel_blob2=False),
    "2Block2D_v_velX+-blob_periodic":     lambda: _make_2block_piso_2d(_R, _R, _R, viscosity=_VI, vel2=(-_BV,0), vel_blob1=True, vel_blob2=True),
    "2Block2D_v_velX+constY+_closed":     lambda: _make_2block_piso_2d(_R, _R, _R, closed_bounds=True, viscosity=_VI, vel_blob1=False, vel_blob2=False),
    # ---- 2D transform ----
    "1Block2D_T-I_velX+const_periodic":   lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), transform_strength=[1,1], vel_blob=False),
    "1Block2D_T-u_velX+const_periodic":   lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), domain_scale=_S2, vel_blob=False),
    "1Block2D_T-n_velX+const_periodic":   lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), domain_scale=_SN2, vel_blob=False),
    "1Block2D_T-s_velX+const_periodic":   lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), domain_scale=[1,2], vel_blob=False),
    "1Block2D_T-e_velX+const_periodic":   lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), transform_strength=[1.05,1.1], vel_blob=False),
    "1Block2D_T-I_velY-blob_periodic":    lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), transform_strength=[1,1], vel_blob=True),
    "1Block2D_T-u_velY-blob_periodic":    lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), domain_scale=_S2, vel_blob=True),
    "1Block2D_T-n_velY-blob_periodic":    lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), domain_scale=_SN2, vel_blob=True),
    "1Block2D_T-s_velY-blob_periodic":    lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), domain_scale=[1,2], vel_blob=True),
    "1Block2D_T-e_velY-blob_periodic":    lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), transform_strength=[1.05,1.1], vel_blob=True),
    "1Block2D_T-I_velX+blob_closed":      lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), closed_bounds=True, transform_strength=[1,1], vel_blob=True),
    "1Block2D_T-u_velX+blob_closed":      lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), closed_bounds=True, domain_scale=_S2, vel_blob=True),
    "1Block2D_T-n_velX+blob_closed":      lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), closed_bounds=True, domain_scale=_SN2, vel_blob=True),
    "1Block2D_T-s_velX+blob_closed":      lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), closed_bounds=True, domain_scale=[1,2], vel_blob=True),
    "1Block2D_T-e_velX+blob_closed":      lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), closed_bounds=True, transform_strength=[1.05,1.1], vel_blob=True),
    # ---- 2D transform + viscosity ----
    "1Block2D_T-I_v_velX+const_periodic": lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, transform_strength=[1,1], vel_blob=False),
    "1Block2D_T-u_v_velX+const_periodic": lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, domain_scale=_S2, vel_blob=False),
    "1Block2D_T-n_v_velX+const_periodic": lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, domain_scale=_SN2, vel_blob=False),
    "1Block2D_T-s_v_velX+const_periodic": lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, domain_scale=[1,2], vel_blob=False),
    "1Block2D_T-e_v_velX+const_periodic": lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, transform_strength=[1.05,1.1], vel_blob=False),
    "1Block2D_T-I_v_velY-blob_periodic":  lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), viscosity=_VI, transform_strength=[1,1], vel_blob=True),
    "1Block2D_T-u_v_velY-blob_periodic":  lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), viscosity=_VI, domain_scale=_S2, vel_blob=True),
    "1Block2D_T-n_v_velY-blob_periodic":  lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), viscosity=_VI, domain_scale=_SN2, vel_blob=True),
    "1Block2D_T-2_v_velY-blob_periodic":  lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), viscosity=_VI, domain_scale=[1,2], vel_blob=True),
    "1Block2D_T-e_v_velY-blob_periodic":  lambda: _make_1block_piso(_R, _R+1, vel=(0,-_BV), viscosity=_VI, transform_strength=[1.05,1.1], vel_blob=True),
    "1Block2D_T-I_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, closed_bounds=True, transform_strength=[1,1], vel_blob=True),
    "1Block2D_T-u_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, closed_bounds=True, domain_scale=_S2, vel_blob=True),
    "1Block2D_T-n_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, closed_bounds=True, domain_scale=_SN2, vel_blob=True),
    "1Block2D_T-s_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, closed_bounds=True, domain_scale=[1,2], vel_blob=True),
    "1Block2D_T-e_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, closed_bounds=True, transform_strength=[1.05,1.1], vel_blob=True),
    # ---- 3D simple ----
    "1Block3D_velX+const_periodic":       lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), vel_blob=False),
    "1Block3D_velX+blob_periodic":        lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), vel_blob=True),
    "1Block3D_velX-blob_periodic":        lambda: _make_1block_piso(_R, _R, _R+2, vel=(-_BV,0,0), vel_blob=True),
    "1Block3D_velY+blob_periodic":        lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,_BV,0), vel_blob=True),
    "1Block3D_velY-blob_periodic":        lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,-_BV,0), vel_blob=True),
    "1Block3D_velZ+blob_periodic":        lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,0,_BV), vel_blob=True),
    "1Block3D_velZ-blob_periodic":        lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,0,-_BV), vel_blob=True),
    "1Block3D_velX+blob_closed":          lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), closed_bounds=True, vel_blob=True),
    # ---- 3D viscosity ----
    "1Block3D_v_velX+const_periodic":     lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), viscosity=_VI, vel_blob=False),
    "1Block3D_v_velX+blob_periodic":      lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), viscosity=_VI, vel_blob=True),
    "1Block3D_v_velX-blob_periodic":      lambda: _make_1block_piso(_R, _R, _R+2, vel=(-_BV,0,0), viscosity=_VI, vel_blob=True),
    "1Block3D_v_velY+blob_periodic":      lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,_BV,0), viscosity=_VI, vel_blob=True),
    "1Block3D_v_velY-blob_periodic":      lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,-_BV,0), viscosity=_VI, vel_blob=True),
    "1Block3D_v_velZ+blob_periodic":      lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,0,_BV), viscosity=_VI, vel_blob=True),
    "1Block3D_v_velZ-blob_periodic":      lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,0,-_BV), viscosity=_VI, vel_blob=True),
    "1Block3D_v_velX+blob_closed":        lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), viscosity=_VI, closed_bounds=True, vel_blob=True),

    # Inflow/outflow ducts, matching the MHD env's boundary configuration.
    "1Block2D_velX+blob_duct":            lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), inflow_outflow=True, vel_blob=True),
    "1Block2D_v_velX+blob_duct":          lambda: _make_1block_piso(_R, _R+1, vel=(_BV,0), viscosity=_VI, inflow_outflow=True, vel_blob=True),
    "1Block3D_velX+blob_duct":            lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), inflow_outflow=True, vel_blob=True),
    "1Block3D_v_velX+blob_duct":          lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), viscosity=_VI, inflow_outflow=True, vel_blob=True),
    # ---- 3D transform + viscosity ----
    "1Block3D_T-I_v_velZ+blob_periodic":  lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,0,_BV), viscosity=_VI, transform_strength=[1,1,1], vel_blob=True),
    "1Block3D_T-u_v_velZ+blob_periodic":  lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,0,_BV), viscosity=_VI, domain_scale=_S3, vel_blob=True),
    "1Block3D_T-e_v_velZ+blob_periodic":  lambda: _make_1block_piso(_R, _R, _R+2, vel=(0,0,_BV), viscosity=_VI, transform_strength=[1.04,1.07,1.1], vel_blob=True),
    "1Block3D_T-I_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), viscosity=_VI, closed_bounds=True, transform_strength=[1,1,1], vel_blob=True),
    "1Block3D_T-u_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), viscosity=_VI, closed_bounds=True, domain_scale=_S3, vel_blob=True),
    "1Block3D_T-e_v_velX+blob_closed":    lambda: _make_1block_piso(_R, _R, _R+2, vel=(_BV,0,0), viscosity=_VI, closed_bounds=True, transform_strength=[1.04,1.07,1.1], vel_blob=True),
}


PISO_TESTS = {
    "SetupAdvectionMatrix":        test_SetupAdvectionMatrix,
    "SetupAdvectionMatrix_scalar": test_SetupAdvectionMatrix_scalar,
    "SetupAdvectionScalar":        test_SetupAdvectionScalar,
    "SetupAdvectionVelocity":      test_SetupAdvectionVelocity,
    "SetupPressureCorrection":     test_SetupPressureCorrection,
    "SetupPressureMatrix":         test_SetupPressureMatrix,
    "SetupPressureRHS":            test_SetupPressureRHS,
    "SetupPressureRHSdiv":         test_SetupPressureRHSdiv,
    "CorrectVelocity":             test_CorrectVelocity,
    "AdvectionSchemeActive":       test_AdvectionSchemeActive,
    "LinearSolve_Scalar":          test_LinearSolve_Scalar,
    "LinearSolve_Velocity":        test_LinearSolve_Velocity,
}

SCHEME_SENSITIVE_TESTS = [
    "SetupAdvectionMatrix",
    "SetupAdvectionMatrix_scalar",
    "SetupAdvectionScalar",
    "SetupAdvectionVelocity",
    "AdvectionSchemeActive",
    "SetupPressureCorrection",
    "SetupPressureRHS",
    "SetupPressureRHSdiv",
    "SetupPressureMatrix",
    "CorrectVelocity",
]


_LINEAR_SOLVE_UNCONVERGED = {
    # Uniform velocity + periodic: no blob structure, purely convective operator
    name for name in PISO_DOMAIN_SETUPS if name.endswith("const_periodic")
} | {
    "1Block2D_T-n_velY-blob_periodic",
}

TEST_DOMAIN_EXCLUSIONS: dict[str, set[str]] = {
    "LinearSolve_Scalar": _LINEAR_SOLVE_UNCONVERGED,
    "LinearSolve_Velocity": _LINEAR_SOLVE_UNCONVERGED,
}


def run_piso_tests(
    test_names: list[str] | None = None,
    domain_names: list[str] | None = None,
    time_step: float = 0.5,
) -> dict[str, str | None]:
    """Run PISO gradcheck tests.

    Parameters
    ----------
    test_names:
        Subset of PISO_TESTS keys to run. Runs all if None.
    domain_names:
        Subset of PISO_DOMAIN_SETUPS keys to use. Runs all if None.
    time_step:
        Time step value passed to each test.

    Returns
    -------
    dict mapping "[scheme] test/domain" → None if the test passed, else the
    failure reason. The reason is kept rather than a bool so the summary and the
    failure report can name what actually went wrong.
    """
    results: dict[str, str | None] = {}
    _test_names = test_names or list(PISO_TESTS.keys())
    _domain_names = domain_names or list(PISO_DOMAIN_SETUPS.keys())

    for test_name in _test_names:
        test_fn = PISO_TESTS[test_name]
        excluded = TEST_DOMAIN_EXCLUSIONS.get(test_name, frozenset())
        for domain_name in _domain_names:
            if domain_name in excluded:
                LOG.info("Skipping %s/%s (excluded)", test_name, domain_name)
                continue
            domain_fn = _with_advection_scheme(PISO_DOMAIN_SETUPS[domain_name])
            label = f"[{ADVECTION_SCHEME}] {test_name}/{domain_name}"
            LOG.info("Running %s ...", label)
            try:
                test_fn(domain_fn, time_step)
                results[label] = None
            except Exception as exc:
                LOG.error("FAILED %s: %s", label, exc)
                results[label] = f"{type(exc).__name__}: {exc}"

    return results


# ============================================================================
# MHD GRADIENT TESTS
# ============================================================================

def _init_epot_fields(domain: PISOtorch.Domain) -> None:
    """Allocate epot tensors and build the Poisson matrix (fixed geometry)."""
    domain.CreateEpotRHS()
    domain.CreateEpotResult()
    domain.CreateEpotMatrix()
    domain.UpdateDomainData()
    PISOtorch.SetupEpotMatrix(domain, 0, False)
    domain.CreateEpotOnBlocks()
    domain.UpdateDomainData()


def make_domain_hartmann(ny: int = 4, nx: int = 8, H: float = 1.0, L: float = 2.0):
    """2D Hartmann domain: closed ±y, periodic x. B along y → e_b = [0,1,0].

    Mirrors the geometry of validation.py (type='hartmann').
    """
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -H), (L, -H), (0.0, H), (L, H)],
        None,
        dtype=DTYPE,
    )
    grid = grid.to(cuda_device).contiguous()
    viscosity = torch.tensor([0.01], dtype=DTYPE, device=cpu_device)
    domain = PISOtorch.Domain(2, viscosity, name="Hartmann", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.MakePeriodic("x")
    domain.PrepareSolve()
    _init_epot_fields(domain)
    # Random velocity so u×B is non-trivial
    block.setVelocity(torch.randn_like(block.velocity))
    domain.UpdateDomainData()
    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    return domain, e_b


def make_domain_hartmann_rot(
    ny: int = 4, nx: int = 8, H: float = 1.0, L: float = 2.0,
    rotate_deg: float = 15.0,
):
    """2D Hartmann domain with rotated (non-orthogonal) grid.
    Matches the hartmann_5_rot.yaml case (rotate_grid_deg=15).
    B along y → e_b = [0,1,0].
    """
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -H), (L, -H), (0.0, H), (L, H)],
        None,
        dtype=DTYPE,
    )
    distance_scaling = shapes.make_rotation_distance_scaling_fn_sine_half(rotate_deg, 1.0)
    grid = shapes.rotate_grid(
        grid,
        angle=rotate_deg,
        distance_scaling=distance_scaling,
        distance_axes=[5.0, 1.0],
    )
    grid = grid.to(cuda_device).contiguous()
    viscosity = torch.tensor([0.01], dtype=DTYPE, device=cpu_device)
    domain = PISOtorch.Domain(2, viscosity, name="HartmannRot", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.MakePeriodic("x")
    domain.PrepareSolve()
    _init_epot_fields(domain)
    block.setVelocity(torch.randn_like(block.velocity))
    domain.UpdateDomainData()
    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    return domain, e_b


def make_domain_shercliff(
    ny: int = 3, nx: int = 5, nz: int = 3, H: float = 1.0, L: float = 1.5
):
    """3D Shercliff/Hunt domain: closed ±y ±z, periodic x. B along y → e_b = [0,1,0].

    Mirrors the geometry of validation.py (type='shercliff' or 'hunt').
    """
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -H), (L, -H), (0.0, H), (L, H)],
        None,
        dtype=DTYPE,
    )
    grid = shapes.extrude_grid_z(grid, res_z=nz, start_z=-H, end_z=H)
    grid = grid.to(cuda_device).contiguous()
    viscosity = torch.tensor([0.01], dtype=DTYPE, device=cpu_device)
    domain = PISOtorch.Domain(3, viscosity, name="Shercliff", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.CloseBoundary("-z")
    block.CloseBoundary("+z")
    block.MakePeriodic("x")
    domain.PrepareSolve()
    _init_epot_fields(domain)
    block.setVelocity(torch.randn_like(block.velocity))
    domain.UpdateDomainData()
    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    return domain, e_b


def make_domain_duct(
    ny: int = 3, nx: int = 5, nz: int = 3, L: float = 1.0, length: float = 2.0
):
    """3D square-duct domain: closed ±y ±z (full duct), periodic x.
    B along y → e_b = [0,1,0].

    Mirrors the geometry of validation_duct.py (duct='full').
    """
    grid = shapes.generate_grid_vertices_2D(
        [2 * ny + 1, nx + 1],
        [(0.0, -L), (length * L, -L), (0.0, L), (length * L, L)],
        None,
        dtype=DTYPE,
    )
    grid = shapes.extrude_grid_z(grid, res_z=2 * nz, start_z=-L, end_z=L)
    grid = grid.to(cuda_device).contiguous()
    viscosity = torch.tensor([0.01], dtype=DTYPE, device=cpu_device)
    domain = PISOtorch.Domain(3, viscosity, name="Duct", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.CloseBoundary("-z")
    block.CloseBoundary("+z")
    block.MakePeriodic("x")
    domain.PrepareSolve()
    _init_epot_fields(domain)
    block.setVelocity(torch.randn_like(block.velocity))
    domain.UpdateDomainData()
    e_b = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    return domain, e_b


def make_domain_temp(
    nx: int = 3, ny: int = 3, nz: int = 3, L: float = 1.0, H: float = 1.0
):
    """3D MHD thermal-convection domain: all walls closed, temperature BCs on ±z.
    B along x → e_b = [1,0,0].

    Mirrors the geometry of validation_temp.py.
    """
    a = L / 2
    half_h = H / 2
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(-a, -half_h), (a, -half_h), (-a, half_h), (a, half_h)],
        None,
        dtype=DTYPE,
    )
    grid = shapes.extrude_grid_z(grid, res_z=nz, start_z=-a, end_z=a)
    grid = grid.to(cuda_device).contiguous()
    viscosity = torch.tensor([0.01], dtype=DTYPE, device=cpu_device)
    domain = PISOtorch.Domain(3, viscosity, name="Temp", device=cuda_device, dtype=DTYPE)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")
    block.CloseBoundary("-x")
    block.CloseBoundary("+x")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.CloseBoundary("-z")
    block.CloseBoundary("+z")
    thermal_diffusivity = torch.tensor([0.01], dtype=DTYPE, device=cuda_device)
    domain.setScalarViscosity(thermal_diffusivity)
    block.getBoundary("-z").setPassiveScalar(
        torch.tensor([[1.0]], dtype=DTYPE, device=cuda_device)
    )
    block.getBoundary("+z").setPassiveScalar(
        torch.tensor([[0.0]], dtype=DTYPE, device=cuda_device)
    )
    domain.PrepareSolve()
    _init_epot_fields(domain)
    block.setVelocity(torch.randn_like(block.velocity))
    domain.UpdateDomainData()
    e_b = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    return domain, e_b


# ----------------------------------------------------------------------------
# MHD gradcheck helpers
# ----------------------------------------------------------------------------

def _copy_epot_returnable(domain: PISOtorch.Domain, epot_result: torch.Tensor):
    """CopyEpotResultToBlocks variant that returns block epots for gradcheck."""

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, er):
            domain.setEpotResult(er)
            domain.CreateEpotOnBlocks()
            domain.UpdateDomainData()
            PISOtorch.CopyEpotResultToBlocks(domain)
            return (*[block.epot.clone() for block in domain.getBlocks()],)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *block_grads):
            if ctx.needs_input_grad[0]:
                for block, g in zip(domain.getBlocks(), block_grads):
                    block.setEpotGrad(g.contiguous())
                domain.CreateEpotResultGrad()
                domain.UpdateDomainData()
                PISOtorch.CopyEpotResultGradFromBlocks(domain)
                return domain.epotResultGrad
            return None

    return _Fn.apply(epot_result)


# ----------------------------------------------------------------------------
# MHD test functions
# ----------------------------------------------------------------------------

def test_ComputeEpotRHS(domain_fn, label: str = "") -> None:
    """Gradcheck ComputeEpotRHS backward against finite differences."""
    domain, e_b = domain_fn()
    dims = domain.getSpatialDims()
    total_size = domain.getTotalSize()

    # Random in-plane u×B field (the in-plane slice is dims * totalSize)
    vec_field = torch.randn(
        dims * total_size, dtype=DTYPE, device=cuda_device, requires_grad=True
    )

    test = torch.autograd.gradcheck(
        lambda vf: PISOtorch_diff.ComputeEpotRHS(domain, vf),
        (vec_field,),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        nondet_tol=GRADCHECK_NONDET_TOL,
    )
    LOG.info("gradcheck ComputeEpotRHS [%s]: %s", label, test)


def test_ComputeCurrentDensityFaceBased(domain_fn, label: str = "") -> None:
    """Gradcheck ComputeCurrentDensityFaceBased backward w.r.t. phi and u×B."""
    domain, e_b = domain_fn()
    total_size = domain.getTotalSize()

    epot = torch.randn(total_size, dtype=DTYPE, device=cuda_device, requires_grad=True)
    # Full 3-component u×B (ComputeCurrentDensityFaceBased always takes 3*totalSize)
    u_cross_eb = torch.randn(
        3 * total_size, dtype=DTYPE, device=cuda_device, requires_grad=True
    )

    test = torch.autograd.gradcheck(
        lambda ep, ucb: PISOtorch_diff.ComputeCurrentDensityFaceBased(domain, ep, ucb),
        (epot, u_cross_eb),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        nondet_tol=GRADCHECK_NONDET_TOL,
    )
    LOG.info("gradcheck ComputeCurrentDensityFaceBased [%s]: %s", label, test)


def test_CopyEpotResultToBlocks(domain_fn, label: str = "") -> None:
    """Gradcheck CopyEpotResultToBlocks backward (epotResult → block.epot)."""
    domain, e_b = domain_fn()
    total_size = domain.getTotalSize()

    epot_result = torch.randn(
        total_size, dtype=DTYPE, device=cuda_device, requires_grad=True
    )

    def fn(er):
        outputs = _copy_epot_returnable(domain, er)
        return torch.cat([o.reshape(-1) for o in outputs])

    test = torch.autograd.gradcheck(
        fn,
        (epot_result,),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        nondet_tol=GRADCHECK_NONDET_TOL,
    )
    LOG.info("gradcheck CopyEpotResultToBlocks [%s]: %s", label, test)


# ----------------------------------------------------------------------------
# Full end-to-end MHD chain test
# ----------------------------------------------------------------------------

def test_mhd_chain(domain_fn, label: str = "") -> None:
    """Gradcheck the full MHD forward chain:
    u×B  →  ComputeEpotRHS  →  (linear_solve handled by autograd)
    phi, u×B  →  ComputeCurrentDensityFaceBased  →  J  →  N*(J×e_b)  →  F_L
    """
    domain, e_b = domain_fn()
    dims = domain.getSpatialDims()
    total_size = domain.getTotalSize()
    eb_dev = e_b.to(cuda_device).to(DTYPE)

    # Inputs: flat velocity field (dims * totalSize)
    vel_flat = torch.randn(
        dims * total_size, dtype=DTYPE, device=cuda_device, requires_grad=True
    )
    epot = torch.randn(total_size, dtype=DTYPE, device=cuda_device, requires_grad=True)
    N = torch.tensor(0.5, dtype=DTYPE, device=cuda_device, requires_grad=True)

    def fn(vf, phi, stuart_n):
        # u × e_b (differentiable pure PyTorch)
        u_cross_eb = _compute_u_cross_eb_flat(vf, eb_dev.cpu(), total_size, dims).to(cuda_device)
        u_cross_eb_inplane = u_cross_eb[: dims * total_size]

        # divergence RHS → phi would be solved here; use fixed phi as input
        _ = PISOtorch_diff.ComputeEpotRHS(domain, u_cross_eb_inplane)

        # current density
        J_flat: torch.Tensor = PISOtorch_diff.ComputeCurrentDensityFaceBased(domain, phi, u_cross_eb)
        J = J_flat.reshape(3, total_size).T  # [totalSize, 3]
        eb_exp = eb_dev.expand_as(J)
        F_L = stuart_n * torch.cross(J, eb_exp, dim=1)  # [totalSize, 3]
        return F_L.reshape(-1)

    test = torch.autograd.gradcheck(
        fn,
        (vel_flat, epot, N),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        nondet_tol=GRADCHECK_NONDET_TOL,
    )
    LOG.info("gradcheck mhd_chain [%s]: %s", label, test)


# ----------------------------------------------------------------------------
# MHD test registry
# ----------------------------------------------------------------------------

MHD_DOMAIN_SETUPS = {
    # key: (factory_fn, factory_kwargs, is_non_ortho)
    "hartmann_2d":     (make_domain_hartmann,     {}, False),
    "hartmann_rot_2d": (make_domain_hartmann_rot, {}, True),   # 15 deg rotation → non-orthogonal
    "shercliff_3d":    (make_domain_shercliff,    {}, False),
    "duct_3d":         (make_domain_duct,         {}, False),
    "temp_3d":         (make_domain_temp,         {}, False),
}

MHD_TESTS = {
    "ComputeEpotRHS":        test_ComputeEpotRHS,
    "ComputeCurrentDensityFaceBased": test_ComputeCurrentDensityFaceBased,
    "CopyEpotResultToBlocks":        test_CopyEpotResultToBlocks,
    "mhd_chain":                     test_mhd_chain,
}


def run_mhd_tests(
    test_names: list[str] | None = None,
    domain_names: list[str] | None = None,
) -> dict[str, str | None]:
    """Run MHD gradcheck tests.

    Parameters
    ----------
    test_names:
        Subset of MHD_TESTS keys to run. Runs all if None.
    domain_names:
        Subset of MHD_DOMAIN_SETUPS keys to use. Uses all if None.

    Returns
    -------
    dict mapping "[scheme] test/domain" → None if the test passed, else the
    failure reason.
    """
    results: dict[str, str | None] = {}
    _test_names = test_names or list(MHD_TESTS.keys())
    _domain_names = domain_names or list(MHD_DOMAIN_SETUPS.keys())

    for test_name in _test_names:
        test_fn = MHD_TESTS[test_name]
        for domain_name in _domain_names:
            domain_fn_cls, domain_kwargs, _is_non_ortho = MHD_DOMAIN_SETUPS[domain_name]
            label = f"[{ADVECTION_SCHEME}] {test_name}/{domain_name}"
            LOG.info("Running %s ...", label)
            try:
                test_fn(
                    _with_advection_scheme(
                        lambda cls=domain_fn_cls, kw=domain_kwargs: cls(**kw)
                    ),
                    label=domain_name,
                )
                results[label] = None
            except Exception as exc:
                LOG.error("FAILED %s: %s", label, exc)
                results[label] = f"{type(exc).__name__}: {exc}"

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import argparse

    parser = argparse.ArgumentParser(description="MHD + PISO gradient checks")
    parser.add_argument(
        "--suite",
        choices=["all", "piso", "mhd"],
        default="all",
        help="Which test suite to run (default: all)",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=None,
        help="Test names to run within the chosen suite (default: all). "
        "PISO choices: " + ", ".join(PISO_TESTS.keys()) + ". "
        "MHD choices: " + ", ".join(MHD_TESTS.keys()),
    )
    parser.add_argument(
        "--piso-domains",
        nargs="*",
        default=None,
        help="PISO domain setups to use (default: all). Choices: "
        + ", ".join(PISO_DOMAIN_SETUPS.keys()),
    )
    parser.add_argument(
        "--mhd-domains",
        nargs="*",
        default=None,
        help="MHD domain setups to use (default: all). Choices: "
        + ", ".join(MHD_DOMAIN_SETUPS.keys()),
    )
    parser.add_argument(
        "--advection-scheme",
        choices=list(ADVECTION_SCHEMES.keys()),
        default="central",
        help="Convective scheme to apply to every domain (default: central). "
        "These are the schemes the solver is run with; both have an adjoint.",
    )
    parser.add_argument(
        "--all-schemes",
        action="store_true",
        help="Re-run the scheme-sensitive tests (" + ", ".join(SCHEME_SENSITIVE_TESTS)
        + ") once per scheme (" + ", ".join(ADVECTION_SCHEMES)
        + "), in addition to the normal run.",
    )
    parser.add_argument(
        "--failures-file",
        default="gradcheck_failures.txt",
        help="Write the failed test cases and their reasons here (default: "
        "gradcheck_failures.txt). Always rewritten, so a run with no failures "
        "leaves an empty report rather than a stale one. Pass '' to skip.",
    )
    args = parser.parse_args()

    results: dict[str, str | None] = {}

    def _run_suites(tests: list[str] | None) -> dict[str, str | None]:
        """Run the requested suites, dispatching each test name to its own suite."""
        out: dict[str, str | None] = {}
        if args.suite in ("all", "piso"):
            piso = [t for t in tests if t in PISO_TESTS] if tests else None
            if piso or tests is None:
                out.update(run_piso_tests(test_names=piso, domain_names=args.piso_domains))
        if args.suite in ("all", "mhd"):
            mhd = [t for t in tests if t in MHD_TESTS] if tests else None
            if mhd or tests is None:
                out.update(run_mhd_tests(test_names=mhd, domain_names=args.mhd_domains))
        return out

    ADVECTION_SCHEME = args.advection_scheme
    results.update(_run_suites(args.tests))

    if args.all_schemes:
        for scheme in ADVECTION_SCHEMES:
            if scheme == args.advection_scheme:
                continue  # already covered by the run above
            ADVECTION_SCHEME = scheme
            LOG.info("=== advection scheme: %s ===", scheme)
            results.update(_run_suites(args.tests or SCHEME_SENSITIVE_TESTS))

    failures = {label: why for label, why in results.items() if why is not None}
    total = len(results)
    passed = total - len(failures)
    print(f"\n{'='*60}")
    for label, why in results.items():
        status = "PASS" if why is None else "FAIL"
        print(f"  [{status}] {label}")
    print(f"{'='*60}")
    print(f"Results: {passed}/{total} passed")

    if args.failures_file:
        import datetime
        import sys

        # Rewritten unconditionally: an empty report is the signal that the last
        # run was clean, otherwise a stale file reads as a current failure list.
        lines = [
            "# gradcheck failures",
            f"# generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"# command:   {' '.join(sys.argv)}",
            f"# result:    {len(failures)} failed of {total}",
            "",
        ]
        for label, why in failures.items():
            lines.append(label)
            for line in str(why).splitlines():
                lines.append(f"    {line}")
            lines.append("")

        with open(args.failures_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        if failures:
            print(f"Wrote {len(failures)} failure(s) to {args.failures_file}")
        else:
            print(f"No failures; wrote empty report to {args.failures_file}")
