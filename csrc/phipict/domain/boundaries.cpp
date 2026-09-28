// Original work Copyright 2025 Aleksandra Franz, Nils Thuerey
// Modified work Copyright 2026 Jannis Becktepe
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Boundary conditions (fixed, Dirichlet, connected, periodic).

#include "domain/domain_internal.h"

std::string Boundary::ToString() const {
	return BoundaryTypeToString(type);
}

std::shared_ptr<const Domain> Boundary::getParentDomain() const {
	if(std::shared_ptr<const Domain> parentDomain = wp_parentDomain.lock()) {
		return parentDomain;
	} else {
		TORCH_CHECK(false, "Parent Domain is expired.");
	}
}

torch::Dtype Boundary::getDtype() const {
	return getParentDomain()->getDtype();
}
torch::Device Boundary::getDevice() const {
	return getParentDomain()->getDevice();
}
index_t Boundary::getSpatialDims() const {
	return getParentDomain()->getSpatialDims();
}
bool Boundary::hasPassiveScalar() const {
	return getParentDomain()->hasPassiveScalar();
}
index_t Boundary::getPassiveScalarChannels() const {
	return getParentDomain()->getPassiveScalarChannels();
}

/** Check spatial size of NCDHW or NDHWC tensors*/
bool checkTensorSpatialSize(torch::Tensor &t, const I4 size, bool channelsFirst){
	const index_t tensorDims = t.dim();
	const index_t offset = channelsFirst ? -1 : -2;
	if(tensorDims<3) return true; //tensor has no spatial dimensions
	const index_t dims = tensorDims-2;
	for(index_t dim=0; dim<dims; ++dim){
		if(t.size(tensorDims + offset - dim)!=size.a[dim]) { return false; }
	}
	return true;
}

I4 getTensorSpatialSize(torch::Tensor &t, bool channelsFirst){
	TORCH_CHECK(t.dim()>2, "Invalid tensor for boundary shape.")
	const index_t offset = channelsFirst ? 1 : 0;
	I4 size = {.a={1,1,1,t.dim()-2}};
	for(index_t dim=0; dim<size.w; ++dim){
		size.a[dim] = t.size(size.w+offset-dim);
	}
	return size;
}

// TODO: finish implementation
// /*
FixedBoundary::FixedBoundary(optional<torch::Tensor> velocity, BoundaryConditionType velocityType,
							// torch::Tensor &pressure, BoundaryConditionType pressureType,
							 optional<torch::Tensor> passiveScalar, optional<BoundaryConditionType> passiveScalarType,
							// const index_t passiveScalarChannels,
							 optional<torch::Tensor> transform, const std::shared_ptr<const Domain> p_parentDomain)
		: Boundary(BoundaryType::FIXED, p_parentDomain) {
	
	// valid formats:
	// static: rank 2, batch + spatialDims (1-3) elements
	// varying: rank 3-5, NCDHW
	
	if(velocity && velocity.value().dim()>2){
		//setSpatialDims(velocity.value().dim()-2);
		setSizeFromTensor(velocity.value(), true);
		//setDtypeDeviceFromTensor(velocity.value());
	} else if(passiveScalar && passiveScalar.value().dim()>2){
		setSizeFromTensor(passiveScalar.value(), true);
	} else if(transform){
		setSizeFromTensor(transform.value(), false);
	} else if(velocity){
		TORCH_CHECK(velocity.value().dim()>1, "Velocity must be 1-3D and either static (shape NC) or varying (shape NCDHW).");
	} else if(passiveScalar){
		TORCH_CHECK(false, "Can't create FixedBoundary from static passive scalar alone.")
	} else {
		TORCH_CHECK(false, "Any field or 'size' is required to create FixedBoundary.");
	}
	
	if(velocity){
		setVelocity(velocity.value());
	} else {
		CreateVelocity(true);
	}
	setVelocityType(velocityType);
	
	/* Pressure boundary is fixed to 0-Neumann, but the fields are needed for correct accessing */
	// if(pressure) ...
	CreatePressure(true);
	setPressureType(BoundaryConditionType::NEUMANN);
	
	if(passiveScalar){
		setPassiveScalar(passiveScalar.value());
	} else if(p_parentDomain->hasPassiveScalar()){
		CreatePassiveScalar(true);
	}
	if(hasPassiveScalar()){
		setPassiveScalarType(passiveScalarType.value_or(m_velocityType));
	}
	
	if(transform){
		setTransform(transform.value());
	}
	
	isTensorChanged=true;
}
/*
FixedBoundary::FixedBoundary(const I4 size, BoundaryConditionType velocityType,
				const index_t passiveScalarChannels, optional<BoundaryConditionType> passiveScalarType,
				const torch::Dtype type, const torch::Device device)
		: Boundary(BoundaryType::FIXED) {
	
	setSpatialDims(size.w);
	setSize(size);
	
	CreateVelocity(true);
	setVelocityType(velocityType);
	
	if(passiveScalarChannels>0){
		CreatePassiveScalar(true);
	}
	setPassiveScalarType(passiveScalarType.value_or(m_velocityType));
	
	isTensorChanged=true;
}
*/

#ifdef DTOR_MSG
FixedBoundary::~FixedBoundary(){
	py::print("FixedBoundary dtor");
	//DetachFwd();
}
#endif

// The FixedBoundary constructor does not take the potential BC, so Copy/Clone must carry
// it over explicitly or a copied domain silently reverts to all-insulating walls.
void FixedBoundary::CopyPotentialBCTo(FixedBoundary &other, const bool cloneTensors) const {
	other.m_potentialType = m_potentialType;
	other.m_potentialCw = m_potentialCw;
	other.m_potentialTypes = cloneTensors ? cloneOptionalTensor(m_potentialTypes) : m_potentialTypes;
	other.m_potentialValues = cloneTensors ? cloneOptionalTensor(m_potentialValues) : m_potentialValues;
}

std::shared_ptr<Boundary> FixedBoundary::Copy() const {
	torch::Tensor v = m_velocity;
	//torch::Tensor p = pressure;
	optional<torch::Tensor> s = m_passiveScalar;
	optional<torch::Tensor> t = m_transform;
	std::shared_ptr<FixedBoundary> newBound = std::make_shared<FixedBoundary>(v, m_velocityType, s, nullopt, t, getParentDomain());
	if(hasPassiveScalar()){ newBound->setPassiveScalarType(m_passiveScalarTypes.value());}
	CopyPotentialBCTo(*newBound, false);
	return newBound;
}

std::shared_ptr<Boundary> FixedBoundary::Clone() const {
	torch::Tensor v = m_velocity.clone();
	//torch::Tensor p = pressure;
	optional<torch::Tensor> s = cloneOptionalTensor(m_passiveScalar);
	optional<torch::Tensor> t = cloneOptionalTensor(m_transform);
	std::shared_ptr<FixedBoundary> newBound = std::make_shared<FixedBoundary>(v, m_velocityType, s, nullopt, t, getParentDomain());
	if(hasPassiveScalar()){ newBound->setPassiveScalarType(m_passiveScalarTypes.value());}
	CopyPotentialBCTo(*newBound, true);
	return newBound;
}

void FixedBoundary::DetachFwd() {
	m_velocity = m_velocity.detach();
	m_pressure = m_pressure.detach();
	if(hasPassiveScalar()) { m_passiveScalar = m_passiveScalar.value().detach(); }
	if(m_transform){ m_transform = m_transform.value().detach(); }
	if(m_potentialValues){ m_potentialValues = m_potentialValues.value().detach(); }
}
void FixedBoundary::DetachGrad() {
#ifdef WITH_GRAD
	m_velocity_grad = m_velocity_grad.detach();
	if(m_passiveScalar_grad) { m_passiveScalar_grad = m_passiveScalar_grad.value().detach(); }
	if(m_potentialValuesGrad) { m_potentialValuesGrad = m_potentialValuesGrad.value().detach(); }
#endif //WITH_GRAD
}
void FixedBoundary::Detach() {
	DetachFwd();
	DetachGrad();
}

void FixedBoundary::setVelocity(torch::Tensor &t){
	index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)>=1, "Batch size (velocity) must be at least 1."); // 1 (shared) or the batch size, checked in UpdateDomain
	TORCH_CHECK(t.size(1)==spatialDims, "Velocity channels are invalid. Velocity must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.dim()==2 || t.dim()==(2+spatialDims),
		"Velocity spatial dimensions are invalid. Velocity must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.scalar_type()==getDtype(), "Velocity dtype must match domain.");
	bool velocityStatic = t.dim()==2;
	if(!velocityStatic && hasSize()){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), true), "New velocity spatial dimensions must match existing fields.");
	}
	
	m_velocity = t;
	m_velocityStatic = velocityStatic;
	
	// keep gradient coherent
	if(!IsTensorEmpty(m_velocity_grad)){
		if(!(m_velocity_grad.dim()==t.dim())){
			// TODO if grad is static: broadcast to varying, else: sum to static
			TORCH_CHECK(false, "New velocity does not match existing gradient tensor.");
		}
	}
	
	isTensorChanged=true;
}

void FixedBoundary::setVelocityType(const BoundaryConditionType velocityType){
	m_velocityType = velocityType;
}

void FixedBoundary::CreateVelocity(const bool createStatic){
	torch::TensorOptions valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	torch::Tensor velocity;
	if(createStatic){
		velocity = torch::zeros({1, getSpatialDims()}, valueOptions);
	} else {
		TORCH_CHECK(hasSize(), "FixedBoundary is missing size to create a varying velocity.");
		const I4 size = getSizes();
		velocity = CreateTensor(1, getSpatialDims(), size, getSpatialDims(), valueOptions);
	}
	setVelocity(velocity);
}

#ifdef WITH_GRAD
void FixedBoundary::setVelocityGrad(torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	index_t spatialDims = getSpatialDims();
	CheckTensor(t, spatialDims, m_velocity, "FixedBoundary.velocity_grad");
	
	m_velocity_grad = t;
	isTensorChanged=true;
}

void FixedBoundary::CreateVelocityGrad(){
	torch::Tensor velocity_grad = torch::zeros_like(m_velocity);
	setVelocityGrad(velocity_grad);
}

void FixedBoundary::setPotentialValuesGrad(const torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	const index_t spatialDims = getSpatialDims();
	TORCH_CHECK(t.dim() == spatialDims + 2 && t.size(1) == 1,
		"Potential values grad must have shape [batch,1,(D),H,W] with size 1 along the face normal.");
	TORCH_CHECK(t.scalar_type() == getDtype(), "Potential values grad must have the domain's dtype.");
	m_potentialValuesGrad = t;
	isTensorChanged = true;
}

#endif //WITH_GRAD

torch::Tensor FixedBoundary::getVelocityVarying(optional<I4> sizeopt) const {
	if(m_velocityStatic){
		TORCH_CHECK(hasSize() || sizeopt, "FixedBoundary has no fields with spatial dimensions to infer varying velocity shape.");
		torch::Tensor vel = m_velocity;
		const I4 size = hasSize() ? getSizes() : sizeopt.value();
		const index_t spatialDims = getSpatialDims();
		std::vector<int64_t> tileMul;
		// NC -> NCDHW
		for(index_t dim=0; dim<spatialDims; ++dim){
			vel = torch::unsqueeze(vel, -1);
			tileMul.push_back(size.a[spatialDims-1-dim]); // z,y,x
		}
		
		vel = torch::tile(vel, tileMul);
		
		return vel.clone().contiguous();
	} else {
		return m_velocity;
	}
}
void FixedBoundary::makeVelocityVarying(optional<I4> size){
	if(m_velocityStatic){
		torch::Tensor velVarying = getVelocityVarying(size);
		setVelocity(velVarying);
	}
}
torch::Tensor FixedBoundary::getVelocity(const bool computational) const {
	TORCH_CHECK(hasSize(), "FixedBoundary has no size set.");
	if(computational && hasTransform()){
		if(m_velocityStatic){
			torch::Tensor velVarying = getVelocityVarying(nullopt);
			return TransformVectors(velVarying, m_transform.value(), true);
		} else {
			return TransformVectors(m_velocity, m_transform.value(), true);
		}
	} else {
		if(m_velocityStatic){
			return getVelocityVarying(nullopt);
		} else {
			return m_velocity;
		}
	}
}

torch::Tensor FixedBoundary::GetFluxes() const {
	TORCH_CHECK(hasSize(), "FixedBoundary has no size set.");
	
	optional<torch::Tensor> fluxes = nullopt;
	if(hasTransform()){
		torch::Tensor vel = getVelocityVarying(nullopt);
		torch::Tensor det = m_transform.value().index({torch::indexing::Ellipsis, -1});
		fluxes = torch::unsqueeze(det, 1) * TransformVectors(vel, m_transform.value(), true);
	} else {
		fluxes = getVelocityVarying(nullopt);
	}
	
	fluxes = fluxes.value().index({torch::indexing::Slice(), torch::indexing::Slice(m_axis,m_axis+1), torch::indexing::Ellipsis});
	
	return fluxes.value().clone();
}

void FixedBoundary::setPressure(torch::Tensor &t){
	index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)>=1, "Batch size (pressure) must be at least 1.");
	TORCH_CHECK(t.size(1)==1, "Pressure channels must be 1.");
	TORCH_CHECK(t.dim()==2 || t.dim()==(2+spatialDims),
		"Pressure spatial dimensions are invalid. Pressure must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.scalar_type()==getDtype(), "Pressure dtype must match domain.");
	bool pressureStatic = t.dim()==2;
	if(!pressureStatic && hasSize()){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), true), "New pressure spatial dimensions must match existing fields.");
	}
	
	m_pressure = t;
	m_pressureStatic = pressureStatic;
	isTensorChanged=true;
}
void FixedBoundary::setPressureType(const BoundaryConditionType pressureType){
	TORCH_CHECK(pressureType==BoundaryConditionType::NEUMANN, "Invalid pressure boundary type: Currently only Neumann boundaries are supported");
	m_pressureType = pressureType;
}
void FixedBoundary::CreatePressure(const bool createStatic){
	torch::TensorOptions valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	torch::Tensor pressure;
	if(createStatic){
		pressure = torch::zeros({1, 1}, valueOptions);
	} else {
		TORCH_CHECK(hasSize(), "FixedBoundary is missing size to create a varying Pressure.");
		const I4 size = getSizes();
		pressure = CreateTensor(1, 1, size, getSpatialDims(), valueOptions);
	}
	setPressure(pressure);
}


void FixedBoundary::setPassiveScalar(torch::Tensor &ps){
	const index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(ps);
	TORCH_CHECK(ps.size(0)>=1, "Batch size (passiveScalar) must be at least 1.");
	TORCH_CHECK(ps.size(1)==getPassiveScalarChannels(), "Passive scalar channels must match domain.");
	TORCH_CHECK(ps.dim()==2 || ps.dim()==(2+spatialDims), "Passive Scalar spatial dimensions are invalid. Passive Scalar must be 1-3D and either static (shape NC) or varying (shape NCDHW). Varying dimensionality must match velocity.");
	TORCH_CHECK(ps.scalar_type()==getDtype(), "Passive Scalar dtype must match domain.");
	bool passiveScalarStatic = ps.dim()==2;
	
	if(!passiveScalarStatic){
		if(hasSize()){
			TORCH_CHECK(checkTensorSpatialSize(ps, getSizes(), true), "New passive Scalar spatial dimensions must match existing fields.");
		} else {
			setSizeFromTensor(ps, true);
		}
	}
	
	m_passiveScalar = ps;
	m_passiveScalarStatic = passiveScalarStatic;
	
	// keep gradient coherent
	if(m_passiveScalar_grad){
		if(!(m_passiveScalar_grad.value().dim()==ps.dim())){
			// TODO if grad is static: broadcast to varying, else: sum to static
			TORCH_CHECK(false, "New passive scalar does not match existing gradient tensor.");
		}
	}
	
	isTensorChanged=true;
}
void FixedBoundary::CreatePassiveScalar(const bool createStatic){
	const index_t passiveScalarChannels = getPassiveScalarChannels();
	torch::TensorOptions valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	torch::Tensor passiveScalar;
	if(createStatic){
		passiveScalar = torch::zeros({1, passiveScalarChannels}, valueOptions);
	} else {
		TORCH_CHECK(hasSize(), "FixedBoundary is missing size to create a varying PassiveScalar.");
		const I4 size = getSizes();
		passiveScalar = CreateTensor(1, passiveScalarChannels, size, getSpatialDims(), valueOptions);
	}
	setPassiveScalar(passiveScalar);
}
void FixedBoundary::setPassiveScalarType(const BoundaryConditionType passiveScalarType) {
	//TORCH_CHECK(passiveScalarType==BoundaryConditionType::DIRICHLET, "Invalid passive scalar boundary type: Currently only Dirichlet boundaries are supported");
	//TORCH_CHECK(passiveScalarType==m_velocityType, "Invalid passive scalar boundary type: Currently must match velocity boundary type.");
	//m_passiveScalarType = passiveScalarType;
	setPassiveScalarType(std::vector<BoundaryConditionType>{passiveScalarType});
}
void FixedBoundary::setPassiveScalarType(const std::vector<BoundaryConditionType> passiveScalarTypes) {
	TORCH_CHECK(hasPassiveScalar(), "FixedBoundary has not passive scalar.");
	TORCH_CHECK(passiveScalarTypes.size()==1 || static_cast<index_t>(passiveScalarTypes.size())==getPassiveScalarChannels(), "Passive scalar boundary conditions must be static or match channels.")

	m_passiveScalarTypes = passiveScalarTypes;

	if(!isPassiveScalarBoundaryTypeStatic()){
		torch::TensorOptions options = torch::TensorOptions().dtype(torch_kBoundaryType).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
		m_passiveScalarTypes_tensor = torch::empty(passiveScalarTypes.size(), options);
		CopyToGPU(m_passiveScalarTypes_tensor.value().data_ptr<BoundaryConditionType_base_type>(), m_passiveScalarTypes.value().data(), passiveScalarTypes.size()*sizeof(BoundaryConditionType));
	}

	isTensorChanged=true;
}
/* void FixedBoundary::clearPassiveScalar() {
	if(hasPassiveScalar()){
		m_passiveScalar = nullopt;
		m_passiveScalarTypes = nullopt;
		m_passiveScalarTypes_tensor=nullopt;
		if(m_passiveScalar_grad){ m_passiveScalar_grad = nullopt; }
		isTensorChanged = true;
	}
} */

#ifdef WITH_GRAD
void FixedBoundary::setPassiveScalarGrad(torch::Tensor &t){
	TORCH_CHECK(m_passiveScalar, "FixedBoundary does not have a passive scalar.");
	CHECK_INPUT_CUDA(t);
	CheckTensor(t, getPassiveScalarChannels(), m_passiveScalar.value(), "FixedBoundary.passiveScalar_grad");
	
	m_passiveScalar_grad = t;
	isTensorChanged=true;
}

void FixedBoundary::CreatePassiveScalarGrad(){
	TORCH_CHECK(m_passiveScalar, "FixedBoundary does not have a passive scalar.");
	torch::Tensor passiveScalar = torch::zeros_like(m_passiveScalar.value());
	setPassiveScalarGrad(passiveScalar);
}

#endif //WITH_GRAD
void FixedBoundary::setTransform(torch::Tensor &t){
	const index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)==1, "Batches are not yet supported (transform).");
	TORCH_CHECK(t.dim()==(2+spatialDims), "Transform spatial dimensions are invalid. Transform must be 1-3D and have shape NDHWT. Dimensionality must match velocity.");
	
	if(hasSize()){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), false), "Transform spatial dimensions must match existing fields.");
	} else {
		setSizeFromTensor(t, false);
	}
	
	TORCH_CHECK(t.size(-1)==TransformNumValues(spatialDims), "Transform channels must match spatial dimensions");
	
	m_transform = t;
	isTensorChanged=true;
}
void FixedBoundary::clearTransform(){
	if(hasTransform()){
		m_transform = nullopt;
		isTensorChanged = true;
	}
}
index_t FixedBoundary::getDim(const dim_t dim) const {
	if(!m_velocityStatic){
		return m_velocity.size(dim);
	} else if(hasPassiveScalar() && !m_passiveScalarStatic){
		return m_passiveScalar.value().size(dim);
	} else if(hasTransform()){
		return m_transform.value().size(dim-1);
	}
	return 1;
}
index_t FixedBoundary::getAxis(const dim_t dim) const {
	TORCH_CHECK(0<=dim && dim<getSpatialDims(), "Axis index must be within spatial dimensions");
	return getDim(getSpatialDims()+1-dim); //velocity.size(velocity.dim()-1 -dim);
}
/* void FixedBoundary::setSpatialDims(const index_t dims) {
	TORCH_CHECK(m_spatialDims==0, "Internal Error: FixedBoundary already has dimensionality set.");
	TORCH_CHECK(0<dims && dims<4, "FixedBoundary must be 1-3D.");
	m_spatialDims = dims;
} */
/*
index_t FixedBoundary::getSpatialDims() const {
	return m_spatialDims;
	// if(m_velocityStatic){
		// return velocity.size(0);
	// }else{
		// return velocity.dim() - 2;
	// }
}*/
bool FixedBoundary::hasSize() const {
	//return !m_velocityStatic || (hasPassiveScalar() && !m_passiveScalarStatic) || hasTransform();
	return m_size.has_value();
}
void FixedBoundary::setSize(I4 size) {
	TORCH_CHECK(!hasSize(), "Internal Error: FixedBoundary already has a size set.");
	size.w = getSpatialDims();
	bool hasSize1 = false; // 1 spatial dimension of a boundary is always 1
	index_t dim = 0; 
	for(;dim<size.w; ++dim){
		if(size.a[dim]==1) {
			TORCH_CHECK(!hasSize1, "Invalid boundary shape: only one dimension must be 1.");
			hasSize1 = true;
			m_axis = dim;
		} else {
			TORCH_CHECK(size.a[dim]>2, "Invalid boundary shape: grid size must be at least 3.");
		}
	}
	for(; dim<3; ++dim){
		size.a[dim]=1;
	}
	TORCH_CHECK(hasSize1, "Invalid boundary shape: one dimension must be 1.");
	m_size = size;
}
void FixedBoundary::setSizeFromTensor(torch::Tensor &t, bool channelsFirst){
	TORCH_CHECK(t.dim()==(getSpatialDims()+2), "Invalid tensor for boundary shape.");
	setSize(getTensorSpatialSize(t, channelsFirst));
}
I4 FixedBoundary::getSizes() const {
	return m_size.value_or(makeI4(1,1,1,getSpatialDims()));
	
	// switch (getSpatialDims())
	// {
	// case 1:
		// return {{.x=getDim(2), .y=1, .z=1, .w=1}};
		// break;
	// case 2:
		// return {{.x=getDim(3), .y=getDim(2), .z=1, .w=2}};
		// break;
	// case 3:
		// return {{.x=getDim(4), .y=getDim(3), .z=getDim(2), .w=3}};
		// break;
	
	// default:
		// return {{.x=1, .y=1, .z=1, .w=1}};
	// }
}
I4 FixedBoundary::getStrides() const {
	const I4 sizes = getSizes();
	return {{.x=1, .y=sizes.x, .z=sizes.x*sizes.y, .w=sizes.x*sizes.y*sizes.z}};
}
/* void FixedBoundary::setDtypeDeviceFromTensor(torch::Tensor &t) {
	TORCH_CHECK(!m_dtype.has_value() && !m_device.has_value(), "FixedBoundary already has device or dtype set.");
	m_device = t.device();
	m_dtype = t.scalar_type();
}
torch::Dtype FixedBoundary::getDtype() const {
	TORCH_CHECK(m_dtype.has_value(), "FixedBoundary does not have a dtype set.");
	return m_dtype.value();
}
torch::Device FixedBoundary::getDevice() const {
	TORCH_CHECK(m_device.has_value(), "FixedBoundary does not have a device set.");
	return m_device.value();
} */
std::string FixedBoundary::ToString() const {
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(";
	repr << getSpatialDims() << "D";
	repr << ", size=" << (hasSize() ? I4toString(getSizes()) : "?");
	repr << ", vStatic=" << (m_velocityStatic ? "true" : "false");
	repr << ", sChannels=" << getPassiveScalarChannels();
	repr << ", sStatic=" << (m_passiveScalarStatic ? "true" : "false");
	repr << ", transform=" << (hasTransform() ? "true" : "false");
	repr << ")";
	return repr.str();
}
//*/

StaticDirichletBoundary::StaticDirichletBoundary(torch::Tensor &slip, torch::Tensor &velocity, torch::Tensor &passiveScalar, const std::shared_ptr<const Domain> p_parentDomain) 
		: Boundary(BoundaryType::DIRICHLET, p_parentDomain), slip(slip), boundaryVelocity(velocity), boundaryScalar(passiveScalar) {
	CHECK_INPUT_HOST(slip);
	TORCH_CHECK(slip.dim()==1, "slip must be 1D.");
	TORCH_CHECK(slip.size(0)==1, "slip must be a scalar.");
	
	CHECK_INPUT_HOST(velocity);
	TORCH_CHECK(velocity.dim()==1, "velocity must be 1D.");
	TORCH_CHECK(0<velocity.size(0) && velocity.size(0)<4, "velocity must have length 1, 2, or 3.");
	
	CHECK_INPUT_HOST(passiveScalar);
	TORCH_CHECK(passiveScalar.dim()==1, "passiveScalar must be 1D.");
	TORCH_CHECK(passiveScalar.size(0)==1, "passiveScalar must have length 1");
	
	TORCH_CHECK(slip.dtype()==velocity.dtype(), "slip and velocity must have same dtype.");
	TORCH_CHECK(passiveScalar.dtype()==velocity.dtype(), "passiveScalar and velocity must have same dtype.");
}
std::shared_ptr<Boundary> StaticDirichletBoundary::Copy() const {
	torch::Tensor s = slip;
	torch::Tensor bv = boundaryVelocity;
	torch::Tensor bs = boundaryScalar;
	return std::make_shared<StaticDirichletBoundary>(s, bv, bs, getParentDomain());
}
std::shared_ptr<Boundary> StaticDirichletBoundary::Clone() const {
	torch::Tensor s = slip.clone();
	torch::Tensor bv = boundaryVelocity.clone();
	torch::Tensor bs = boundaryScalar.clone();
	return std::make_shared<StaticDirichletBoundary>(s, bv, bs, getParentDomain());
}
index_t StaticDirichletBoundary::getSpatialDims() const {
    return boundaryVelocity.size(0);
}

std::string StaticDirichletBoundary::ToString() const {
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(" << getSpatialDims() << "D)";
	return repr.str();
}

VaryingDirichletBoundary::VaryingDirichletBoundary(torch::Tensor &slip, torch::Tensor &velocity, torch::Tensor &passiveScalar, const std::shared_ptr<const Domain> p_parentDomain)
		: Boundary(BoundaryType::DIRICHLET_VARYING, p_parentDomain), slip(slip), boundaryVelocity(velocity), boundaryScalar(passiveScalar) {
	CHECK_INPUT_HOST(slip);
	TORCH_CHECK(slip.dim()==1, "slip must be 1D.");
	TORCH_CHECK(slip.size(0)==1, "slip must be a scalar.");
	
	CHECK_INPUT_CUDA(velocity);
	index_t spatialDims = velocity.dim() - 2;
	TORCH_CHECK(spatialDims>0 && spatialDims<4, "Only 1D, 2D, and 3D is supported. layout should be NC<spatial-dims>, e.g. NCDHW for 3D.");
	TORCH_CHECK(velocity.size(1)==spatialDims, "The velocity channels must match the spatial dimensions.");
	TORCH_CHECK(velocity.size(0)==1, "Batches are not yet supported (velocity).");
	//numDims = spatialDims + 2;
	
	CHECK_INPUT_CUDA(passiveScalar);
	TORCH_CHECK(passiveScalar.dim()==velocity.dim(), "passiveScalar must have same dimensionality as velocity.");
	TORCH_CHECK(passiveScalar.size(1)==1, "The passiveScalar channels must be 1.");
	TORCH_CHECK(passiveScalar.size(0)==1, "Batches are not yet supported (passiveScalar).");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==passiveScalar.size(dim), "spatial dimension must match");
	}
	
	TORCH_CHECK(slip.dtype()==velocity.dtype(), "slip and velocity must have same dtype.");
	TORCH_CHECK(passiveScalar.dtype()==velocity.dtype(), "passiveScalar and velocity must have same dtype.");
}
std::shared_ptr<Boundary> VaryingDirichletBoundary::Copy() const {
	torch::Tensor s = slip;
	torch::Tensor bv = boundaryVelocity;
	torch::Tensor bs = boundaryScalar;
	std::shared_ptr<VaryingDirichletBoundary> newBound = std::make_shared<VaryingDirichletBoundary>(s, bv, bs, getParentDomain());
	if(hasTransform){
		torch::Tensor t = transform;
		newBound->setTransform(t);
	}
	return newBound;
}
std::shared_ptr<Boundary> VaryingDirichletBoundary::Clone() const {
	torch::Tensor s = slip.clone();
	torch::Tensor bv = boundaryVelocity.clone();
	torch::Tensor bs = boundaryScalar.clone();
	std::shared_ptr<VaryingDirichletBoundary> newBound = std::make_shared<VaryingDirichletBoundary>(s, bv, bs, getParentDomain());
	if(hasTransform){
		torch::Tensor t = transform.clone();
		newBound->setTransform(t);
	}
	return newBound;
}
void VaryingDirichletBoundary::setTransform(torch::Tensor &newTransform) {
	CHECK_INPUT_CUDA(newTransform);
	TORCH_CHECK(newTransform.dim()==boundaryVelocity.dim(), "new Transform dimensions must match velocity.");
	TORCH_CHECK(newTransform.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform channels must match spatial dimensions");
	for(int dim=2;dim<boundaryVelocity.dim();++dim){
		TORCH_CHECK(boundaryVelocity.size(dim)==newTransform.size(dim-1), "spatial dimension must match");
	}
	TORCH_CHECK(newTransform.dtype()==boundaryVelocity.dtype(), "Transform and velocity must have same dtype.");
	
	transform = newTransform;
	hasTransform = true;
	isTensorChanged=true;
}
void VaryingDirichletBoundary::clearTransform(){
	hasTransform = false;
	transform = torch::empty(0);
	isTensorChanged=true;
}
/*
bool VaryingDirichletBoundary::CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const std::string &name) const {
	return CheckTensor(tensor, channels, boundaryVelocity, name);
}
torch::Tensor VaryingDirichletBoundary::CreateDataTensor(const index_t channels) const {
	return CreateTensor();
}*/
void VaryingDirichletBoundary::setVelocity(torch::Tensor &t){
	CheckTensor(t, getSpatialDims(), boundaryVelocity, "Velocity");
	boundaryVelocity = t;
	isTensorChanged=true;
}
torch::Tensor VaryingDirichletBoundary::getVelocity(const bool computational) const {
	if(computational && hasTransform){
		return TransformVectors(boundaryVelocity, transform, true);
	}
	return boundaryVelocity;
}
torch::Tensor VaryingDirichletBoundary::getBoundaryFlux() const {
	//torch::Tensor boundaryFluxes = hasTransform ? ComputeFluxes(boundaryVelocity, transform) : boundaryVelocity.clone();
	
	return torch::empty(0);
}
void VaryingDirichletBoundary::setPassiveScalar(torch::Tensor &t){
	CheckTensor(t, 1, boundaryScalar, "PassiveScalar");
	boundaryScalar = t;
	isTensorChanged=true;
}

#ifdef WITH_GRAD
void VaryingDirichletBoundary::setVelocityGrad(torch::Tensor &t){
	CheckTensor(t, getSpatialDims(), boundaryVelocity, "VelocityGrad");
	boundaryVelocity_grad = t;
	isTensorChanged=true;
}
void VaryingDirichletBoundary::CreateVelocityGrad(){
	boundaryVelocity_grad = CreateTensorFromRef(1, getSpatialDims(), boundaryVelocity);
	isTensorChanged=true;
}
void VaryingDirichletBoundary::setPassiveScalarGrad(torch::Tensor &t){
	CheckTensor(t, 1, boundaryScalar, "PassiveScalarGrad");
	boundaryScalar_grad = t;
	isTensorChanged=true;
}
void VaryingDirichletBoundary::CreatePassiveScalarGrad(){
	boundaryScalar_grad = CreateTensorFromRef(1, 1, boundaryScalar);
	isTensorChanged=true;
}
#endif

index_t VaryingDirichletBoundary::getSpatialDims() const {
    return boundaryVelocity.dim() - 2;
}

index_t VaryingDirichletBoundary::getDim(const dim_t dim) const {
    return boundaryVelocity.size(dim);
}
index_t VaryingDirichletBoundary::getAxis(const dim_t dim) const {
	TORCH_CHECK(0<=dim && dim<getSpatialDims(), "Axis index must be within spatial dimensions");
    return boundaryVelocity.size(boundaryVelocity.dim()-1 -dim);
}

I4 VaryingDirichletBoundary::getSizes() const {
	switch (getSpatialDims())
	{
	case 1:
		return {{.x=getDim(2), .y=1, .z=1, .w=1}};
		break;
	case 2:
		return {{.x=getDim(3), .y=getDim(2), .z=1, .w=2}};
		break;
	case 3:
		return {{.x=getDim(4), .y=getDim(3), .z=getDim(2), .w=3}};
		break;
	
	default:
		return {{.x=1, .y=1, .z=1, .w=1}};
	}
}

I4 VaryingDirichletBoundary::getStrides() const {
    const I4 sizes = getSizes();
	return {{.x=1, .y=sizes.x, .z=sizes.x*sizes.y, .w=sizes.x*sizes.y*sizes.z}};
}

std::string VaryingDirichletBoundary::ToString() const {
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(" << getSpatialDims() << "D";
	repr << ", size=" << I4toString(getSizes()) << ", stride=" << I4toString(getStrides());
	repr << ")";
	return repr.str();
}

ConnectedBoundary::ConnectedBoundary(std::weak_ptr<Block> wp_connectedBlock, std::vector<dim_t> &axes, const std::shared_ptr<const Domain> p_parentDomain)
		: Boundary(BoundaryType::CONNECTED_GRID, p_parentDomain), axes(axes), wp_connectedBlock(wp_connectedBlock) {
	
	std::shared_ptr<Block> connectedBlock = getConnectedBlock();
	
	TORCH_CHECK(connectedBlock->getSpatialDims()==static_cast<index_t>(axes.size()), "axes must match spatial dimensions of connectedBlock.");
	const index_t numBounds = connectedBlock->getSpatialDims()*2;
	TORCH_CHECK(0<=axes[0] && axes[0]<numBounds, "axes[0] must be in [0,dim*2].");
	
	if(connectedBlock->getSpatialDims()>1){
		TORCH_CHECK(0<=axes[1] && axes[1]<numBounds, "axes[1] must be in [0,dim*2].");
		TORCH_CHECK(BoundaryIndexToDim(axes[0]) != BoundaryIndexToDim(axes[1]), "axes[1] must be a different axis than axes[0].");
		
		if(connectedBlock->getSpatialDims()>2){
			TORCH_CHECK(0<=axes[2] && axes[2]<numBounds, "axes[2] must be in [0,dim*2].");
			TORCH_CHECK(BoundaryIndexToDim(axes[0]) != BoundaryIndexToDim(axes[2]), "axes[2] must be a different axis than axes[0].");
			TORCH_CHECK(BoundaryIndexToDim(axes[1]) != BoundaryIndexToDim(axes[2]), "axes[2] must be a different axis than axes[1].");
		}
	}
}
std::shared_ptr<Block> ConnectedBoundary::getConnectedBlock() const {
	if(std::shared_ptr<Block> connectedBlock = wp_connectedBlock.lock()) {
		return connectedBlock;
	} else {
		TORCH_CHECK(false, "ConnectedBlock is expired.");
	}
}
//torch::Dtype ConnectedBoundary::getDtype() const {return getConnectedBlock()->getDtype();}
//index_t ConnectedBoundary::getSpatialDims() const {return getConnectedBlock()->getSpatialDims();}

dim_t ConnectedBoundary::getConnectionAxis(const dim_t axis) const {
	TORCH_CHECK(0<=axis && axis<getSpatialDims(), "axis out of bounds.")
	return BoundaryIndexToDim(axes[axis]);
}
dim_t ConnectedBoundary::getConnectionAxisDirection(const dim_t axis) const {
	TORCH_CHECK(0<=axis && axis<getSpatialDims(), "axis out of bounds.")
	return axes[axis]&1;
}

std::string ConnectedBoundary::ToString() const {
	
	std::shared_ptr<Block> connectedBlock = getConnectedBlock();
	
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(to=\"" << connectedBlock->name << "\", face=" << BoundaryIndexToString(axes[0]);
	for(index_t dim=1; dim<connectedBlock->getSpatialDims(); ++dim){
		repr << ", axis" << dim << "=" << BoundaryIndexToString(axes[dim]);
	}
	repr << ")";
	return repr.str();
}

/* I4 ConnectedBoundary::getConnectionVector(){
	I4 connections = makeI4();
	return connections;
} */

/* Create a connection between 2 blocks
directional face specification: [-x,+x,-y,+y,-z,+z] <-> [0,5]
=> axis := face/2 ([x,y,z] <-> [0,2])
=> direction := face%2, 0 is lower/negative side, 1 is upper/positive side
	for "connectedAxis" the direction indicates if the connection is inverted (0 for same direction, 1 for inverted)

face1 of block1 is connected to face2 of block2. for 2D and 3D, the remaining axes are also mapped.
for block1:
	face1 connects to face2
	axis[(face1 / 2 + 1)%dims] is aligned to connectedAxis1. The connection is inverted if connectedAxis1%2==1.
	axis[(face1 / 2 + 2)%dims] is aligned to connectedAxis2.
*/
void ConnectBlocks(std::shared_ptr<Block> block1, const dim_t face1, std::shared_ptr<Block> block2, const dim_t face2, const dim_t connectedAxis1, const dim_t connectedAxis2){
	TORCH_CHECK(block1->getParentDomain()==block2->getParentDomain(), "The blocks must belong to the same domain.");
	const dim_t spatialDims = block1->getSpatialDims();
	TORCH_CHECK(spatialDims==block2->getSpatialDims(), "The spatial dimensions of the blocks must match.");
	
	std::vector<dim_t> axes1 = {face2};
	std::vector<dim_t> axes2 = {face1};
	if(spatialDims>1){
		axes1.push_back(connectedAxis1);
		const dim_t face1Dim = BoundaryIndexToDim(face1);
		const dim_t face2Dim = BoundaryIndexToDim(face2);
		bool axesSwapped = false;
		if(spatialDims==2 || (BoundaryIndexToDim(connectedAxis1) == (face2Dim + 1)%spatialDims)) {
			axes2.push_back((((face1Dim + 1)%spatialDims)<<1) | (connectedAxis1&1));
			axesSwapped = false;
		} else {
			//spatialDims==3 here
			TORCH_CHECK((connectedAxis2>>1) == (face2Dim+ 1)%spatialDims, "Invalid connection.")
			axes2.push_back((((face1Dim + 2)%spatialDims)<<1) | (connectedAxis2&1));
			axesSwapped = true;
		}
		if(spatialDims>2){
			axes1.push_back(connectedAxis2);
			if(!axesSwapped){
				axes2.push_back((((face1Dim + 2)%spatialDims)<<1) | (connectedAxis2&1));
			} else {
				axes2.push_back((((face1Dim + 1)%spatialDims)<<1) | (connectedAxis1&1));
			}
		}
	}
	std::shared_ptr<ConnectedBoundary> bound1 = std::make_shared<ConnectedBoundary>(block2, axes1, block1->getParentDomain());
	block1->setBoundary(face1, bound1);
	std::shared_ptr<ConnectedBoundary> bound2 = std::make_shared<ConnectedBoundary>(block1, axes2, block2->getParentDomain());
	block2->setBoundary(face2, bound2);
}
void ConnectBlocks(std::shared_ptr<Block> block1, const std::string &face1, std::shared_ptr<Block> block2, const std::string &face2, const std::string &connectedAxis1, const std::string &connectedAxis2){
	ConnectBlocks(block1, BoundarySideToIndex(face1), block2, BoundarySideToIndex(face2), BoundarySideToIndex(connectedAxis1), BoundarySideToIndex(connectedAxis2));
}

static void CheckPotentialCw(const double cw){
	TORCH_CHECK(cw > 0.0, "A THIN_WALL potential BC needs a wall conductance ratio cw > 0.");
}

void FixedBoundary::setPotentialBC(const PotentialBC type, optional<double> cw){
	if(type == PotentialBC::THIN_WALL){
		TORCH_CHECK(cw.has_value(), "A THIN_WALL potential BC needs a wall conductance ratio cw.");
		CheckPotentialCw(cw.value());
		m_potentialCw = cw.value();
	} else {
		TORCH_CHECK(!cw.has_value() || cw.value() == 0.0, "cw is only valid for a THIN_WALL potential BC.");
		m_potentialCw = 0.0;
	}
	m_potentialType = type;
	if(m_potentialTypes.has_value()){ clearPotentialTypes(); }
}

void FixedBoundary::setPotentialTypes(const torch::Tensor &types, optional<double> cw){
	const index_t spatialDims = getSpatialDims();
	TORCH_CHECK(types.dim() == spatialDims + 2 && types.size(0) == 1 && types.size(1) == 1,
		"Potential BC types must have shape [1,1,(D),H,W] with size 1 along the face normal.");
	index_t numSize1 = 0;
	for(index_t d = 2; d < types.dim(); ++d){ if(types.size(d) == 1) ++numSize1; }
	TORCH_CHECK(numSize1 >= 1, "Potential BC types must have size 1 along the face normal.");
	TORCH_CHECK(!types.is_floating_point() && types.scalar_type() != torch::kBool,
		"Potential BC types must be an integer tensor of PotentialBC values.");
	TORCH_CHECK(types.numel() > 0, "Potential BC types must not be empty.");
	const int64_t minType = types.min().item<int64_t>();
	const int64_t maxType = types.max().item<int64_t>();
	TORCH_CHECK(minType >= static_cast<int64_t>(PotentialBC::INSULATING) && maxType <= static_cast<int64_t>(PotentialBC::THIN_WALL),
		"Potential BC types contain values that are not PotentialBC members.");

	const bool anyThinWall = (types == static_cast<int64_t>(PotentialBC::THIN_WALL)).any().item<bool>();
	if(anyThinWall){
		TORCH_CHECK(cw.has_value(), "Potential BC types contain THIN_WALL cells but no wall conductance ratio cw was given.");
		CheckPotentialCw(cw.value());
		m_potentialCw = cw.value();
	} else {
		TORCH_CHECK(!cw.has_value() || cw.value() == 0.0, "cw is only valid if some cells are THIN_WALL.");
		m_potentialCw = 0.0;
	}
	m_potentialTypes = types.to(getDevice(), torch::kInt8).contiguous();
	// the uniform type is only a fallback once a mask is set; keep it meaningful for readers
	m_potentialType = static_cast<PotentialBC>(minType == maxType ? minType : static_cast<int64_t>(m_potentialType));
	isTensorChanged = true;
}

void FixedBoundary::setPotentialValues(const torch::Tensor &values){
	const index_t spatialDims = getSpatialDims();
	TORCH_CHECK(values.dim() == spatialDims + 2 && values.size(1) == 1,
		"Potential values must have shape [1 or batch,1,(D),H,W] with size 1 along the face normal.");
	TORCH_CHECK(values.is_floating_point(), "Potential values must be a floating point tensor.");
	m_potentialValues = values.to(getDevice(), getDtype()).contiguous();
	isTensorChanged = true;
}

bool FixedBoundary::hasPotentialDirichlet() const {
	if(m_potentialTypes.has_value()){
		return (m_potentialTypes.value() == static_cast<int64_t>(PotentialBC::DIRICHLET)).any().item<bool>();
	}
	return m_potentialType == PotentialBC::DIRICHLET;
}

bool FixedBoundary::hasPotentialThinWall() const {
	if(m_potentialTypes.has_value()){
		return (m_potentialTypes.value() == static_cast<int64_t>(PotentialBC::THIN_WALL)).any().item<bool>();
	}
	return m_potentialType == PotentialBC::THIN_WALL;
}

// Legacy flag setters. The old flags were independent booleans; every combination any
// caller used maps onto exactly one PotentialBC, and clearing a flag returns to the default.
void FixedBoundary::setEpotCw(double cw) {
	TORCH_CHECK(cw >= 0.0, "Epot wall conductance Cw must be non-negative.");
	if(cw > 0.0){
		setPotentialBC(PotentialBC::THIN_WALL, cw);
	} else if(m_potentialType == PotentialBC::THIN_WALL){
		setPotentialBC(PotentialBC::INSULATING, nullopt);
	}
}

void FixedBoundary::setEpotDirichlet(bool d) {
	if(d){
		setPotentialBC(PotentialBC::DIRICHLET, nullopt);
	} else if(m_potentialType == PotentialBC::DIRICHLET){
		setPotentialBC(PotentialBC::INSULATING, nullopt);
	}
}

void FixedBoundary::setEpotInsulating(bool insulating) {
	if(!insulating){
		setPotentialBC(PotentialBC::OPEN, nullopt);
	} else if(m_potentialType == PotentialBC::OPEN){
		setPotentialBC(PotentialBC::INSULATING, nullopt);
	}
}


