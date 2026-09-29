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
// Blocks: fields, boundaries and grid of one structured block.

#include "domain/domain_internal.h"

Block::Block(optional<torch::Tensor> velocity, optional<torch::Tensor> pressure, optional<torch::Tensor> passiveScalar,
		optional<torch::Tensor> vertexCoordinates, const std::string &name, const std::shared_ptr<const Domain> p_parentDomain) : name(name), wp_parentDomain(p_parentDomain) {
	
	TORCH_CHECK(velocity || pressure || passiveScalar || vertexCoordinates, "At least one field (velocity, pressure, passiveScalar, vertexCoordinates) or the size must be given to create a block.")
	
	const index_t spatialDims = p_parentDomain->getSpatialDims();
	//const index_t passiveScalarChannels = p_parentDomain->getPassiveScalarChannels();
	const torch::Dtype dtype = p_parentDomain->getDtype();
	const torch::Device device = p_parentDomain->getDevice();
	
	//index_t spatialDims = 0;
	if(velocity){
		torch::Tensor velocityTensor = velocity.value();
		CHECK_INPUT_CUDA(velocityTensor);
		//spatialDims = velocityTensor.dim() - 2;
		TORCH_CHECK(velocityTensor.dim()==spatialDims+2, "Velocity dimensionality must match domain. Layout should be NC<spatial-dims>, e.g. NCDHW for 3D.");
		TORCH_CHECK(velocityTensor.size(1)==spatialDims, "The velocity channels must match the spatial dimensions.");
		TORCH_CHECK(velocityTensor.size(0)>=1, "Batch size must be at least 1.");
		for(int dim=2;dim<velocityTensor.dim();++dim){
			TORCH_CHECK(velocityTensor.size(dim)>2, "all spatial dimensions must be at least 3.");
		}
		//numDims = spatialDims + 2;
		
		this->velocity = velocityTensor;
	} else {
		TensorInfo fieldInfo;
		if(pressure){
			fieldInfo = getFieldInfo(pressure.value(), false);
		} else if(passiveScalar){
			fieldInfo = getFieldInfo(passiveScalar.value(), false);
		} else {
			fieldInfo = getFieldInfo(vertexCoordinates.value(), true);
		}
		
		TORCH_CHECK(fieldInfo.spatialDims==spatialDims, "Field dimensionality must match domain. Layout should be NC<spatial-dims>, e.g. NCDHW for 3D.");
		//spatialDims = fieldInfo.spatialDims;
		//numDims = fieldInfo.dims;
		
		this->velocity = CreateTensor(1, spatialDims, fieldInfo.size, spatialDims, 
			torch::TensorOptions().dtype(fieldInfo.dtype).layout(torch::kStrided).device(fieldInfo.device.value().type(), fieldInfo.device.value().index()));
	}
	
	// set simple default boundaries before anything might access them.
	boundaries.reserve(spatialDims*2);
	for(index_t bound=0; bound<spatialDims*2 ;++bound){
		boundaries.push_back(std::make_shared<PeriodicBoundary>(p_parentDomain));
	}
	
	if(pressure){
		setPressure(pressure.value());
	}else{
		CreatePressure();
	}
	
	if(passiveScalar){
		setPassiveScalar(passiveScalar.value());
	} else if(p_parentDomain->hasPassiveScalar()){
		CreatePassiveScalar();
	}
	
	if(vertexCoordinates){
		setVertexCoordinates(vertexCoordinates.value());
	}
};

//Block::Block(const I4 size, const index_t passiveScalarChannels, const std::string &name, const torch::Dtype dtype, const torch::Device device) : name(name){
Block::Block(const I4 size, const std::string &name, const std::shared_ptr<const Domain> p_parentDomain) : name(name), wp_parentDomain(p_parentDomain) {
	const index_t spatialDims = p_parentDomain->getSpatialDims(); //size.w;
	//const index_t passiveScalarChannels = p_parentDomain->getPassiveScalarChannels();
	const torch::Dtype dtype = p_parentDomain->getDtype();
	const torch::Device device = p_parentDomain->getDevice();
	
	//numDims = spatialDims + 2;
	TORCH_CHECK(0<spatialDims && spatialDims<4, "Only 1D, 2D, and 3D is supported.");
	for(index_t dim=0; dim<spatialDims; ++dim){
		TORCH_CHECK(size.a[dim]>2, "all spatial dimensions must be at least 3.");
	}
	//TORCH_CHECK(device.is_cuda(), "Device must be a CUDA.")
	
	velocity = CreateTensor(1, spatialDims, size, spatialDims, torch::TensorOptions().dtype(dtype).layout(torch::kStrided).device(device.type(), device.index()));
	
	CreatePressure();
	
	if(p_parentDomain->hasPassiveScalar()){
		CreatePassiveScalar();
	}
	
	boundaries.reserve(spatialDims*2);
	for(index_t bound=0; bound<spatialDims*2 ;++bound){
		boundaries.push_back(std::make_shared<PeriodicBoundary>(p_parentDomain));
	}
}

#ifdef DTOR_MSG
Block::~Block(){
	py::print("Block dtor");
	//DetachFwd();
}
#endif

std::shared_ptr<Block> Block::Copy() const {
	torch::Tensor v = velocity;
	torch::Tensor p = pressure;
	optional<torch::Tensor> s = passiveScalar;
	optional<torch::Tensor> c = m_vertexCoordinates;
	const bool hasOnlyTransform = !m_vertexCoordinates && m_transform;
	std::string n = name+"_copy";
	std::shared_ptr<Block> cBlock = std::make_shared<Block>(v, p, s, c, n, getParentDomain());
	if(hasOnlyTransform){
		// cBlock boundaries are default periodic here, so no boundary transform update happens.
		torch::Tensor t = m_transform.value();
		cBlock->setTransform(t, m_faceTransform);
	}
	if(hasVelocitySource()){
		torch::Tensor vs = velocitySource.value();
		cBlock->setVelocitySource(vs);
	}
	if(hasViscosity()){
		torch::Tensor visc = m_viscosity.value();
		cBlock->setViscosity(visc);
	}
	// copy prescibed boudaries. connected is handled on domain level, periodic is default.
	for(index_t i=0;i<static_cast<index_t>(boundaries.size());++i){
		BoundaryType bt = boundaries[i]->type;
		if(bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::FIXED){
			cBlock->setBoundary(i, boundaries[i]->Copy());
		}
	}
	return cBlock;
}
std::shared_ptr<Block> Block::Clone() const {
	torch::Tensor v = velocity.clone();
	torch::Tensor p = pressure.clone();
	optional<torch::Tensor> s = cloneOptionalTensor(passiveScalar);
	optional<torch::Tensor> c = cloneOptionalTensor(m_vertexCoordinates);
	const bool hasOnlyTransform = !m_vertexCoordinates && m_transform;
	std::string n = name+"_clone";
	std::shared_ptr<Block> cBlock = std::make_shared<Block>(v, p, s, c, n, getParentDomain());
	if(hasOnlyTransform){
		// cBlock boundaries are default periodic here, so no boundary transform update happens.
		torch::Tensor t = m_transform.value().clone();
		cBlock->setTransform(t, cloneOptionalTensor(m_faceTransform));
	}
	if(hasVelocitySource()){
		torch::Tensor vs = velocitySource.value().clone();
		cBlock->setVelocitySource(vs);
	}
	if(hasViscosity()){
		torch::Tensor visc = m_viscosity.value().clone();
		cBlock->setViscosity(visc);
	}
	// copy prescibed boudaries. connected is handled on domain level, periodic is default.
	for(index_t i=0;i<static_cast<index_t>(boundaries.size());++i){
		BoundaryType bt = boundaries[i]->type;
		if(bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::FIXED){
			cBlock->setBoundary(i, boundaries[i]->Clone());
		}
	}
	return cBlock;
}


std::shared_ptr<const Domain> Block::getParentDomain() const {
	if(std::shared_ptr<const Domain> parentDomain = wp_parentDomain.lock()) {
		return parentDomain;
	} else {
		TORCH_CHECK(false, "Parent Domain is expired.");
	}
}

void Block::DetachFwd() {
	velocity = velocity.detach();
	if(velocitySource){ velocitySource = velocitySource.value().detach(); }
	pressure = pressure.detach();
	if(hasPassiveScalar()) { passiveScalar = passiveScalar.value().detach(); }
	if(m_vertexCoordinates){ m_vertexCoordinates = m_vertexCoordinates.value().detach(); }
	if(m_transform){ m_transform = m_transform.value().detach(); }
	if(m_faceTransform){ m_faceTransform = m_faceTransform.value().detach(); }
	if(hasEpot()) { epot = epot.value().detach(); }

	for(const auto &bound : getFixedBoundaries()){
		bound.second->DetachFwd();
	}
}
void Block::DetachGrad() {
	velocity_grad = velocity_grad.detach();
	if(velocitySource_grad){ velocitySource_grad = velocitySource_grad.value().detach(); }
	if(!IsTensorEmpty(pressure_grad)){ pressure_grad = pressure_grad.detach(); }
	if(hasPassiveScalar()) { passiveScalar_grad = passiveScalar_grad.detach(); }
	if(hasEpotGrad()) { epot_grad = epot_grad.value().detach(); }

	for(const auto &bound : getFixedBoundaries()){
		bound.second->DetachGrad();
	}
}
void Block::Detach() {
	DetachFwd();
	DetachGrad();
}


torch::Dtype Block::getDtype() const{
	return getParentDomain()->getDtype();
}
torch::Device Block::getDevice() const{
	return getParentDomain()->getDevice();
}

torch::TensorOptions Block::getValueOptions() const {
	//return torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	return getParentDomain()->getValueOptions();
}

bool Block::CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const bool allowStatic, const std::string &name) const {
	CHECK_INPUT_CUDA(tensor);
	const index_t numDims = getSpatialDims() + 2;
	TORCH_CHECK(tensor.dim()==numDims || (allowStatic && tensor.dim()==2), "Dimensions of " + name + " must be " + std::to_string(numDims) + (allowStatic ? " or 2." : "."));
	// state fields must have the batch size of the block; static-capable data (velocity source,
	// viscosity) may have batch size 1 and is then shared by all environments
	TORCH_CHECK(tensor.size(0)==getBatchSize() || (allowStatic && tensor.size(0)==1),
		"Batch size (dim 0) of " + name + " must be " + std::to_string(getBatchSize()) + (allowStatic ? " or 1." : "."));
	if(channels<1){
		// free number of channels
	}else{
		TORCH_CHECK(tensor.size(1)==channels, "Channels (dim 1) of " + name + " must be " + std::to_string(channels) + ".");
	}
	if(tensor.dim()!=2){
		for(int dim=2;dim<numDims;++dim){
			TORCH_CHECK(velocity.size(dim)==tensor.size(dim), "Spatial dimension " + std::to_string(dim) + " of " + name +
				" must match (" + std::to_string(velocity.size(dim)) + "), but is (" + std::to_string(tensor.size(dim)) + ").");
		}
	}
	TORCH_CHECK(tensor.dtype()==velocity.dtype(), name + "has wrong dtype.");
	return true;
}

torch::Tensor Block::CreateDataTensor(const index_t channels) const {
	
	return CreateTensor(getBatchSize(), channels, getSizes(), getSpatialDims(), getValueOptions());
}

void Block::setVelocity(torch::Tensor &v){
	CheckDataTensor(v, getSpatialDims(), false, "Velocity");
	velocity = v;
	isTensorChanged=true;
}
torch::Tensor Block::getVelocity(const bool computational) const {
	if(computational){
		TORCH_CHECK(hasTransform(), "Coordinates or Transformations are required to compute computational velocities.");
		return TransformVectors(velocity, m_transform.value(), true);
	}
	return velocity;
}

void Block::setVelocitySource(torch::Tensor &t){
	/* index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)==1, "Batches are not yet supported (velocity source).");
	TORCH_CHECK(t.size(1)==spatialDims, "Velocity source channels are invalid. Velocity source must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.dim()==2 || t.dim()==(2+spatialDims),
		"Velocity source spatial dimensions are invalid. Velocity source must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	bool velocityStatic = t.dim()==2;
	if(!velocityStatic){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), true), "New velocity spatial dimensions must match existing fields.");
	} */
	
	CheckDataTensor(t, getSpatialDims(), true, "VelocitySource");
	
	velocitySource = t;
	velocitySourceStatic = t.dim()==2; //velocityStatic;
	
	// keep gradient coherent. The grad is only scratch space of the backward pass (every
	// backward recreates it with zeros_like(velocitySource) before its kernel and reads it out
	// right after), so a stale one of the other shape is dropped. That happens when a hook
	// switches between a static and a varying source within a step (e.g. the channel forcing
	// and the MHD Lorentz force) and a checkpoint replays the forward during backward.
	if(velocitySource_grad){
		if(!(velocitySource_grad.value().dim()==t.dim())){
			velocitySource_grad = nullopt;
		}
	}

	isTensorChanged = true;
}
void Block::CreateVelocitySource(const bool createStatic){
	torch::TensorOptions valueOptions = getValueOptions();
	torch::Tensor velocity;
	if(createStatic){
		velocity = torch::zeros({1, getSpatialDims()}, valueOptions);
	} else {
		const I4 size = getSizes();
		velocity = CreateTensor(1, getSpatialDims(), size, getSpatialDims(), valueOptions);
	}
	setVelocitySource(velocity);
}
void Block::clearVelocitySource(){
	if(hasVelocitySource()){
		velocitySource = nullopt;
		isTensorChanged = true;
	}
}


void Block::setViscosity(const torch::Tensor &v){
	CheckDataTensor(v, 1, true, "BlockViscosity");
	
	if(m_viscosity_grad){
		if(!(m_viscosity_grad.value().dim()==v.dim())){
			// TODO if grad is static: broadcast to varying, else: sum to static
			TORCH_CHECK(false, "New block viscosity does not match existing gradient tensor.");
		}
	}
	
	m_viscosity = v;
	isTensorChanged = true;
}
void Block::clearViscosity(){
	if(hasViscosity()){
		m_viscosity = nullopt;
		isTensorChanged = true;
	}
	
}

void Block::setPressure(torch::Tensor &p){
	CheckDataTensor(p, 1, false, "Pressure");
	pressure = p;
	isTensorChanged=true;
}
void Block::setPassiveScalar(torch::Tensor &s){
	const index_t channels = getPassiveScalarChannels();
	TORCH_CHECK(channels>0, "Passive scalars are not active.");
	CheckDataTensor(s, channels, false, "Passive Scalar");
	passiveScalar = s;
	isTensorChanged=true;
}
void Block::CreatePassiveScalar(){
	const index_t channels = getPassiveScalarChannels();
	TORCH_CHECK(channels>0, "Passive scalars are not active.");
	torch::Tensor sg = CreateDataTensor(channels);
	setPassiveScalar(sg);
}
/* void Block::clearPassiveScalar(){
	if(passiveScalar){
		passiveScalar = nullopt;
		isTensorChanged=true;
	}
} */
/* template <typename scalar_t>
scalar_t* Block::getPassiveScalarDataPtr() const {
	return getOptionalTensorDataPtr(passiveScalar);
} */

bool Block::hasPassiveScalar() const {
	return getParentDomain()->hasPassiveScalar();
}
index_t Block::getPassiveScalarChannels() const {
	//return hasPassiveScalar() ? passiveScalar.value().size(1) : 0;
	return getParentDomain()->getPassiveScalarChannels();
}

void Block::CreatePressure(){
	torch::Tensor p = CreateDataTensor(1);
	setPressure(p);
}
void Block::setEpot(torch::Tensor &e){
	CheckDataTensor(e, 1, false, "Epot");
	epot = e;
	isTensorChanged=true;
}
void Block::CreateEpot(){
	torch::Tensor e = CreateDataTensor(1);
	setEpot(e);
}
void Block::CreateVelocity(){
	torch::Tensor v = CreateDataTensor(getSpatialDims());
	setVelocity(v);
}

#ifdef WITH_GRAD
void Block::setVelocityGrad(torch::Tensor &vg){
	CheckDataTensor(vg, getSpatialDims(), false, "VelocityGrad");
	velocity_grad = vg;
	isTensorChanged=true;
}
void Block::CreateVelocityGrad(){
	torch::Tensor vg = CreateDataTensor(getSpatialDims());
	setVelocityGrad(vg);
}

void Block::setVelocitySourceGrad(torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(velocitySource.has_value(), "Block does not have a velocity source tensor");
	index_t spatialDims = getSpatialDims();
	CheckTensor(t, spatialDims, velocitySource.value(), "Block.velocitySource_grad");
	
	velocitySource_grad = t;
	isTensorChanged=true;

}
void Block::CreateVelocitySourceGrad(){
	if(velocitySource){
		torch::Tensor velocity_grad = torch::zeros_like(velocitySource.value());
		setVelocitySourceGrad(velocity_grad);
	} else {
		clearVelocitySourceGrad();
	}
}
void Block::clearVelocitySourceGrad(){
	if(hasVelocitySourceGrad()){
		velocitySource_grad = nullopt;
		isTensorChanged = true;
	}
}

void Block::setViscosityGrad(const torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(m_viscosity.has_value(), "Block does not have a viscosity tensor");
	index_t spatialDims = getSpatialDims();
	CheckTensor(t, 1, m_viscosity.value(), "Block.viscosity_grad");
	
	m_viscosity_grad = t;
	isTensorChanged=true;
}
void Block::CreateViscosityGrad(){
	if(m_viscosity){
		torch::Tensor viscosity_grad = torch::zeros_like(m_viscosity.value());
		setViscosityGrad(viscosity_grad);
	} else {
		clearViscosityGrad();
	}
}
void Block::clearViscosityGrad(){
	if(hasViscosityGrad()){
		m_viscosity_grad = nullopt;
		isTensorChanged = true;
	}
}

void Block::setPressureGrad(torch::Tensor &pg){
	CheckDataTensor(pg, 1, false, "PressureGrad");
	pressure_grad= pg;
	isTensorChanged=true;
}
void Block::CreatePressureGrad(){
	torch::Tensor pg = CreateDataTensor(1);
	setPressureGrad(pg);
}
void Block::setPassiveScalarGrad(torch::Tensor &sg){
	CheckDataTensor(sg, getPassiveScalarChannels(), false, "PassiveScalarGrad");
	passiveScalar_grad = sg;
	isTensorChanged=true;
}
void Block::CreatePassiveScalarGrad(){
	torch::Tensor sg = CreateDataTensor(getPassiveScalarChannels());
	setPassiveScalarGrad(sg);
}


void Block::setEpotGrad(torch::Tensor &eg){
	epot_grad = eg;
	isTensorChanged=true;
}
void Block::CreateEpotGrad(){
	torch::Tensor eg = CreateDataTensor(1);
	setEpotGrad(eg);
}

void Block::CreatePassiveScalarGradOnBoundaries(){
	for(auto boundary : boundaries){
		if(boundary->type==BoundaryType::FIXED){
			std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
			bound->CreatePassiveScalarGrad();
		}
	}
}
void Block::CreateVelocityGradOnBoundaries(){
	for(auto boundary : boundaries){
		if(boundary->type==BoundaryType::FIXED){
			std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
			bound->CreateVelocityGrad();
		}
	}
}

#endif //WITH_GRAD

torch::Tensor Block::getMaxVelocity(const bool withBounds, const bool computational) const{
	torch::Tensor maxVel = torch::max(torch::abs(getVelocity(computational)));
	
	if(withBounds){
		for(auto boundary : boundaries){
			switch(boundary->type){
				case BoundaryType::DIRICHLET:
				{
					std::shared_ptr<StaticDirichletBoundary> bound = std::static_pointer_cast<StaticDirichletBoundary> (boundary);
					maxVel = torch::maximum(maxVel, torch::max(torch::abs(bound->boundaryVelocity)));
					break;
				}
				case BoundaryType::DIRICHLET_VARYING:
				{
					std::shared_ptr<VaryingDirichletBoundary> bound = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
					maxVel = torch::maximum(maxVel, torch::max(torch::abs(bound->getVelocity(computational))));
					break;
				}
				case BoundaryType::FIXED:
				{
					std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
					maxVel = torch::maximum(maxVel, torch::max(torch::abs(bound->getVelocity(computational))));
					break;
				}
				default:
					break;
			}
		}
	}
	
	return maxVel;
}

/* Maximum absolute value of every batched environment: [B] from data of batch size B, or of
 * batch size 1 (shared by all environments), broadcast. */
static torch::Tensor maxAbsPerEnv(const torch::Tensor &t){
	return torch::abs(t).reshape({t.size(0), -1}).amax(1);
}

torch::Tensor Block::getMaxVelocityPerEnv(const bool withBounds, const bool computational) const{
	torch::Tensor maxVel = maxAbsPerEnv(getVelocity(computational));
	
	if(withBounds){
		for(auto boundary : boundaries){
			switch(boundary->type){
				case BoundaryType::DIRICHLET:
				{
					// no batch dimension: shared by all environments
					std::shared_ptr<StaticDirichletBoundary> bound = std::static_pointer_cast<StaticDirichletBoundary> (boundary);
					maxVel = torch::maximum(maxVel, torch::max(torch::abs(bound->boundaryVelocity)));
					break;
				}
				case BoundaryType::DIRICHLET_VARYING:
				{
					std::shared_ptr<VaryingDirichletBoundary> bound = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
					maxVel = torch::maximum(maxVel, maxAbsPerEnv(bound->getVelocity(computational)));
					break;
				}
				case BoundaryType::FIXED:
				{
					std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
					maxVel = torch::maximum(maxVel, maxAbsPerEnv(bound->getVelocity(computational)));
					break;
				}
				default:
					break;
			}
		}
	}
	
	return maxVel;
}

torch::Tensor getVelocityMagnitude(const torch::Tensor &velocity){ //, const index_t spatialDims, const index_t totalSize){
	//return torch::linalg::vector_norm(velocity, 2, 1, true, c10::nullopt);
	return at::linalg_vector_norm(velocity, 2, 1, true);
}

torch::Tensor Block::getMaxVelocityMagnitude(const bool withBounds, const bool computational) const{
	torch::Tensor maxMag = torch::max(getVelocityMagnitude(getVelocity(computational)));
	
	if(withBounds){
		for(auto boundary : boundaries){
			switch(boundary->type){
				case BoundaryType::DIRICHLET:
				{
					std::shared_ptr<StaticDirichletBoundary> bound = std::static_pointer_cast<StaticDirichletBoundary> (boundary);
					maxMag = torch::maximum(maxMag, torch::max(getVelocityMagnitude(bound->boundaryVelocity)));
					break;
				}
				case BoundaryType::DIRICHLET_VARYING:
				{
					std::shared_ptr<VaryingDirichletBoundary> bound = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
					maxMag = torch::maximum(maxMag, torch::max(getVelocityMagnitude(bound->getVelocity(computational))));
					break;
				}
				case BoundaryType::FIXED:
				{
					std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
					maxMag = torch::maximum(maxMag, torch::max(getVelocityMagnitude(bound->getVelocity(computational))));
					break;
				}
				default:
					break;
			}
		}
	}
	
	return maxMag;
}


bool Block::IsValidFaceIndex(const index_t face) const {
	return 0 <= face && face < (getSpatialDims()*2);
}
bool Block::IsValidAxisIndex(const index_t axis) const {
	return 0 <= axis && axis < getSpatialDims();
}
void Block::CheckFaceIndex(const index_t face) const {
	TORCH_CHECK(IsValidFaceIndex(face), "Face index must be in [0, dims*2) or one of {-x, +x, -y, +y, -z, +z} for exising spatial dimensions.");
}
void Block::CheckAxisIndex(const index_t axis) const {
	TORCH_CHECK(IsValidAxisIndex(axis), "Axis index must be in [0, dims) or one of {x, y, z} for exising spatial dimensions.");
}

void Block::setBoundary(const std::string &side, std::shared_ptr<Boundary> boundary){
    setBoundary(BoundarySideToIndex(side), boundary);
};

void Block::setBoundary(const index_t index, std::shared_ptr<Boundary> boundary){
	CheckFaceIndex(index);
	switch(boundary->type){
	case BoundaryType::FIXED:
		{
			std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
			const index_t spatialDims = getSpatialDims();
			TORCH_CHECK(spatialDims==bound->getSpatialDims(), "Spatial dimensions of block and boundary must match.");
			TORCH_CHECK(getPassiveScalarChannels()==bound->getPassiveScalarChannels(), "Passive scalar channels of block and boundary must match.");
			TORCH_CHECK(getDtype()==bound->getDtype(), "Dtype of block and boundary must match.");
			TORCH_CHECK(getDevice()==bound->getDevice(), "Device of block and boundary must match.");
			if(bound->hasSize()){
				index_t axis = (index>>1);
				TORCH_CHECK(bound->getAxis(axis)==1, "Boundary axis size must be 1.");
				if(spatialDims>1){
					axis = ((index>>1) + 1)%spatialDims;
					TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "Boundary size must match block.");
					if(spatialDims>2){
						axis = ((index>>1) + 2)%spatialDims;
						TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "Boundary size must match block.");
					}
				}
			} else {
				I4 boundSize = getSizes();
				boundSize.a[index>>1] = 1;
				bound->setSize(boundSize);
			}
			if(hasTransform()!=bound->hasTransform()){
				py::print("Warning: Only one of Block and FixedBoundary has a transformation set.");
			}
		}
		break;
	case BoundaryType::DIRICHLET:
		{
			std::shared_ptr<StaticDirichletBoundary> bound = std::static_pointer_cast<StaticDirichletBoundary> (boundary);
			TORCH_CHECK(getSpatialDims()==bound->getSpatialDims(), "Spatial dimensions of block and boundary must match.");
			TORCH_CHECK(getDtype()==bound->getDtype(), "Dtype of block and boundary must match.");
		}
		break;
	case BoundaryType::DIRICHLET_VARYING:
		{
			std::shared_ptr<VaryingDirichletBoundary> bound = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
			const index_t spatialDims = getSpatialDims();
			TORCH_CHECK(spatialDims==bound->getSpatialDims(), "Spatial dimensions of block and boundary must match.");
			TORCH_CHECK(getDtype()==bound->getDtype(), "Dtype of block and boundary must match.");
			//check each dimension/axis
			index_t axis = (index>>1);
			TORCH_CHECK(bound->getAxis(axis)==1, "");
			if(spatialDims>1){
				axis = ((index>>1) + 1)%spatialDims;
				TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "");
				if(spatialDims>2){
					axis = ((index>>1) + 2)%spatialDims;
					TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "");
				}
			}
			if(hasTransform()!=bound->hasTransform){
				py::print("Warning: Only one of Block and VaryingDirichletBoundary has a transformation set.");
			}
		}
		break;
	case BoundaryType::CONNECTED_GRID:
		{
			std::shared_ptr<ConnectedBoundary> bound = std::static_pointer_cast<ConnectedBoundary> (boundary);
			std::shared_ptr<Block> otherBlock = bound->getConnectedBlock();
			const index_t spatialDims = getSpatialDims();
			TORCH_CHECK(spatialDims==otherBlock->getSpatialDims(), "Spatial dimensions of blocks must match.");
			TORCH_CHECK(getDtype()==otherBlock->getDtype(), "Dtype of blocks must match.");
			
			//axis index: 0,1,2 -> x,y,z
			// getDim: 3D [0,4] -> NCzyx
			//const index_t maxDim = spatialDims + 1;
			if(spatialDims>1){
				dim_t axisIndex = ((index>>1)+1)%spatialDims;
				dim_t otherAxisIndex = bound->getConnectionAxis(1);
				TORCH_CHECK(getAxis(axisIndex)==otherBlock->getAxis(otherAxisIndex), "First connection axis size does not match.")
				if(spatialDims>2){
					axisIndex = ((index>>1)+2)%spatialDims;
					otherAxisIndex = bound->getConnectionAxis(2);
					TORCH_CHECK(getAxis(axisIndex)==otherBlock->getAxis(otherAxisIndex), "Second connection axis size does not match.")
				}
			}
		}
		break;
	default:
		break;
	}
	boundaries[index] = boundary;
}

std::shared_ptr<Boundary> Block::getBoundary(const std::string &side) const {
    return getBoundary(BoundarySideToIndex(side));
}

std::shared_ptr<Boundary> Block::getBoundary(const index_t index) const {
	TORCH_CHECK(index>=0 && index<getSpatialDims()*2, "Invalid boundary location specified.");
    std::shared_ptr<Boundary> bound = boundaries[index];
	return bound;
}

std::vector<std::pair<index_t, std::shared_ptr<FixedBoundary>>> Block::getFixedBoundaries() const {
	std::vector<std::pair<index_t, std::shared_ptr<FixedBoundary>>> bounds;
	for(index_t boundIdx=0; boundIdx<(getSpatialDims()*2); ++boundIdx){
		if(boundaries[boundIdx]->type==BoundaryType::FIXED){
			std::pair<index_t, std::shared_ptr<FixedBoundary>> bound = std::make_pair(boundIdx, std::static_pointer_cast<FixedBoundary>(boundaries[boundIdx]));
			bounds.push_back(bound);
		}
	}
	return bounds;
}

bool Block::isAllFixedBoundariesPassiveScalarTypeStatic() const {
	for(const auto &bound : getFixedBoundaries()){
		if(!bound.second->isPassiveScalarBoundaryTypeStatic()){
			return false;
		}
	}
	return true;
}

void Block::CloseConnectedBoudary(const index_t face, const bool useFixed){
	CheckFaceIndex(face);
	std::shared_ptr<Boundary> boundary = getBoundary(face);
	switch(boundary->type){
		case BoundaryType::CONNECTED_GRID:
		{
			std::shared_ptr<ConnectedBoundary> bound = std::static_pointer_cast<ConnectedBoundary> (boundary);
			std::shared_ptr<Block> otherBlock = bound->getConnectedBlock();
			const index_t otherBoundIndex = bound->axes[0];
			// std::shared_ptr<Boundary> otherboundary = otherBlock->getBoundary(otherBoundIndex);
			// if(otherboundary->type == BoundaryType::CONNECTED_GRID){
				// std::shared_ptr<ConnectedBoundary> otherBound = std::static_pointer_cast<ConnectedBoundary> (boundary);
				// if(otherBound->getConnectedBlock() == shared_from_this() && otherBound->axes[0] == static_cast<dim_t>(face)){
					if(useFixed){
						otherBlock->MakeFixedBoundary(otherBoundIndex, nullopt, BoundaryConditionType::DIRICHLET, nullopt, nullopt);
					} else {
						otherBlock->MakeClosedBoundary(otherBoundIndex);
					}
				// }
			// }
			break;
		}
		case BoundaryType::PERIODIC:
		{
			if(useFixed){
				MakeFixedBoundary(face ^ 1, nullopt, BoundaryConditionType::DIRICHLET, nullopt, nullopt);
			}else {
				MakeClosedBoundary(face ^ 1);
			}
			break;
		}
		default:
			break;
	}
}

torch::Tensor Block::GetFaceTransformBoundarySlice(const index_t face) const {
	CheckFaceIndex(face);
	TORCH_CHECK(hasFaceTransform(), "faceTransform missing to slice for boundary.");
	const index_t axis = face >> 1;
	const bool isUpper = face & 1;
	
	std::vector<torch::indexing::TensorIndex> slicing;
	slicing.push_back(torch::indexing::Slice()); // batch, no change
	//slicing.push_back(torch::indexing::Slice(axis, axis+1)); // channel, get for axis, but keep dimension
	slicing.push_back(axis); // channel, get for axis
	for(index_t dim=getSpatialDims()-1; dim>=0; --dim){
		if(dim==axis){
			if(isUpper){
				slicing.push_back(torch::indexing::Slice(-1,torch::indexing::None));
			} else {
				slicing.push_back(torch::indexing::Slice(0,1));
			}
		} else {
			slicing.push_back(torch::indexing::Slice(0,-1)); // remove excess top from staggeted grid
		}
	}
	slicing.push_back(torch::indexing::Slice()); // transform struct size, no change
	
	torch::Tensor boundaryTransform = m_faceTransform.value().index(slicing).clone().contiguous();
	
	return boundaryTransform;
}

void Block::UpdateBoundaryTransforms(){
	if(!hasFaceTransform()) { return; }
	for(index_t bound=0; bound<(getSpatialDims()*2); ++bound){
		std::shared_ptr<Boundary> boundary = getBoundary(bound);
		switch(boundary->type){
			case BoundaryType::DIRICHLET:
			{
				TORCH_CHECK(false, "TODO: Dirichlet->DirichletVarying on transform update?");
				break;
			}
			case BoundaryType::DIRICHLET_VARYING:
			{
				std::shared_ptr<VaryingDirichletBoundary> vdb = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
				torch::Tensor boundaryTransform = GetFaceTransformBoundarySlice(bound);
				vdb->setTransform(boundaryTransform);
				break;
			}
			case BoundaryType::FIXED:
			{
				std::shared_ptr<FixedBoundary> fb = std::static_pointer_cast<FixedBoundary> (boundary);
				torch::Tensor boundaryTransform = GetFaceTransformBoundarySlice(bound);
				fb->setTransform(boundaryTransform);
				break;
			}
			default:
				break;
		}
	}
}

void Block::MakeClosedBoundary(const index_t bound){
	TORCH_CHECK(false, "Old boundary formats are no longer supported.");
	CheckFaceIndex(bound);
	torch::TensorOptions CPUValueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided);
	torch::Tensor boundarySlip = torch::zeros({1}, CPUValueOptions);
	if(hasFaceTransform()){
		torch::TensorOptions valueOptions = getValueOptions();
		const index_t spatialDims = getSpatialDims();
		const index_t boundAxis = bound>>1;
		I4 boundarySize = getSizes();
		boundarySize.a[boundAxis] = 1;
		
		torch::Tensor boundaryVelocity = CreateTensor(1, spatialDims, boundarySize, spatialDims, valueOptions);
		torch::Tensor boundaryScalar = CreateTensor(1, getPassiveScalarChannels(), boundarySize, spatialDims, valueOptions);
		
		std::shared_ptr<VaryingDirichletBoundary> vdb = std::make_shared<VaryingDirichletBoundary>(boundarySlip, boundaryVelocity, boundaryScalar, getParentDomain());
		
		torch::Tensor boundaryTransform = GetFaceTransformBoundarySlice(bound);
		vdb->setTransform(boundaryTransform);
		
		setBoundary(bound, vdb);
	} else {
		torch::Tensor boundaryVelocity = torch::zeros({getSpatialDims()}, CPUValueOptions);
		torch::Tensor boundaryScalar = torch::zeros({getPassiveScalarChannels()}, CPUValueOptions);
		
		std::shared_ptr<StaticDirichletBoundary> sdb = std::make_shared<StaticDirichletBoundary>(boundarySlip, boundaryVelocity, boundaryScalar, getParentDomain());
		setBoundary(bound, sdb);
	}
}
/*
void Block::MakeFixedBoundary(const index_t face,
		optional<torch::Tensor> velocity, const BoundaryConditionType velocityType,
		optional<torch::Tensor> passiveScalar, optional<BoundaryConditionType> scalarType) {
	MakeFixedBoundary(face, velocity, velocityType, passiveScalar, scalarType ? {scalarType} : nullopt);
}*/

void Block::MakeFixedBoundary(const index_t face,
		optional<torch::Tensor> velocity, const BoundaryConditionType velocityType,
		optional<torch::Tensor> passiveScalar, optional<std::vector<BoundaryConditionType>> scalarType) {
	CheckFaceIndex(face);
	
	if(velocity || passiveScalar || hasFaceTransform()){
		optional<torch::Tensor> boundaryTransform = nullopt;
		if(hasFaceTransform()){
			boundaryTransform = GetFaceTransformBoundarySlice(face);
		}
		std::shared_ptr<FixedBoundary> fb = std::make_shared<FixedBoundary>(velocity, velocityType, passiveScalar, nullopt, boundaryTransform, getParentDomain());
		if(scalarType) fb->setPassiveScalarType(scalarType.value());
		setBoundary(face, fb);
	} else {
		torch::Tensor boundaryVelocity = torch::zeros({1, getSpatialDims()}, getValueOptions()); // static (NC), needed to indicate dimensionality
		std::shared_ptr<FixedBoundary> fb = std::make_shared<FixedBoundary>(boundaryVelocity, velocityType, nullopt, nullopt, nullopt, getParentDomain());
		if(scalarType) fb->setPassiveScalarType(scalarType.value());
		setBoundary(face, fb);
	}
}

void Block::ConnectBlock(const dim_t face1, std::shared_ptr<Block> block2, const dim_t face2, const dim_t connectedAxis1, const dim_t connectedAxis2){
	// TODO: check if connection already exists
	CheckFaceIndex(face1);
	CloseConnectedBoudary(face1, true);
	block2->CheckFaceIndex(face2);
	block2->CloseConnectedBoudary(face2, true);
	ConnectBlocks(shared_from_this(), face1, block2, face2, connectedAxis1, connectedAxis2);
}
void Block::ConnectBlock(const std::string &face1, std::shared_ptr<Block> block2, const std::string &face2, const std::string &connectedAxis1, const std::string &connectedAxis2){
	ConnectBlock(BoundarySideToIndex(face1), block2, BoundarySideToIndex(face2), BoundarySideToIndex(connectedAxis1), BoundarySideToIndex(connectedAxis2));
}

void Block::MakePeriodic(const index_t axis){
	CheckAxisIndex(axis);
	const index_t faceLower = axis << 1;
	const index_t faceUpper = faceLower | 1;
	std::shared_ptr<Boundary> boundaryLower = getBoundary(faceLower);
	std::shared_ptr<Boundary> boundaryUpper = getBoundary(faceUpper);
	if(!(boundaryLower->type==BoundaryType::PERIODIC && boundaryUpper->type==BoundaryType::PERIODIC)){
		if(boundaryLower->type==BoundaryType::CONNECTED_GRID){ CloseConnectedBoudary(faceLower, true); }
		if(boundaryUpper->type==BoundaryType::CONNECTED_GRID){ CloseConnectedBoudary(faceUpper, true); }
		
		if(boundaryLower->type!=BoundaryType::PERIODIC){
			std::shared_ptr<PeriodicBoundary> pbLower = std::make_shared<PeriodicBoundary>(getParentDomain());
			setBoundary(faceLower, pbLower);
		}
		if(boundaryUpper->type!=BoundaryType::PERIODIC){
			std::shared_ptr<PeriodicBoundary> pbLower = std::make_shared<PeriodicBoundary>(getParentDomain());
			setBoundary(faceUpper, pbLower);
		}
	}
}
void Block::MakePeriodic(const std::string &axis){
	MakePeriodic(AxisToIndex(axis));
}
void Block::MakeAllPeriodic() {
	for(index_t dim=0; dim<getSpatialDims(); ++dim){
		MakePeriodic(dim);
	}
}

void Block::CloseBoundary(const index_t bound){
	CheckFaceIndex(bound);
	CloseConnectedBoudary(bound, false);
	MakeClosedBoundary(bound);
}
void Block::CloseBoundary(const std::string &face){
	CloseBoundary(BoundarySideToIndex(face));
}
void Block::CloseAllBoundaries(){
	for(index_t face=0; face<getSpatialDims()*2; ++face){
		CloseBoundary(face, nullopt, nullopt);
	}
}

void Block::CloseBoundary(const index_t bound, optional<torch::Tensor> velocity, optional<torch::Tensor> passiveScalar) {
	CheckFaceIndex(bound);
	CloseConnectedBoudary(bound, true);
	MakeFixedBoundary(bound, velocity, BoundaryConditionType::DIRICHLET, passiveScalar, nullopt);
}
void Block::CloseBoundary(const std::string & face, optional<torch::Tensor> velocity, optional<torch::Tensor> passiveScalar) {
	CloseBoundary(BoundarySideToIndex(face), velocity, passiveScalar);
}

void Block::OpenBoundary(const index_t bound, optional<torch::Tensor> passiveScalar) {
	CheckFaceIndex(bound);
	CloseConnectedBoudary(bound, true);
	optional<std::vector<BoundaryConditionType>> scalarType = nullopt;
	if(!passiveScalar.has_value()){
		scalarType = std::vector<BoundaryConditionType>{BoundaryConditionType::NEUMANN};
	}
	MakeFixedBoundary(bound, nullopt, BoundaryConditionType::NEUMANN, passiveScalar, scalarType);
}
void Block::OpenBoundary(const std::string &face, optional<torch::Tensor> passiveScalar) {
	OpenBoundary(BoundarySideToIndex(face), passiveScalar);
}

bool Block::IsUnconnectedBoundary(const index_t index) const {
    BoundaryType bt = boundaries.at(index)->type;
	return bt==BoundaryType::FIXED || bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::NEUMANN;
}
bool Block::hasPrescribedBoundary() const {
	for(auto bound : boundaries){
		BoundaryType bt = bound->type;
		if(bt==BoundaryType::FIXED || bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::NEUMANN){
			return true;
		}
	}
	return false;
}
void Block::setVertexCoordinates(torch::Tensor &newVertexCoordinates){
	CHECK_INPUT_CUDA(newVertexCoordinates);
	TORCH_CHECK(newVertexCoordinates.dim()==velocity.dim(), "new vertex coordinates dimensions must match velocity.");
	TORCH_CHECK(newVertexCoordinates.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newVertexCoordinates.size(1)==getSpatialDims(), "vertex coordinates channel dimension must match spatial dimensions.");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==(newVertexCoordinates.size(dim)-1), "vertex coordinates spatial dimension must match velocity's +1.");
	}
	TORCH_CHECK(newVertexCoordinates.dtype()==velocity.dtype(), "Vertex coordinates and velocity must have same dtype.");
	
	clearCoordsTransforms();
	m_vertexCoordinates = newVertexCoordinates;
	m_transform = CoordsToTransforms(newVertexCoordinates);
	m_faceTransform = CoordsToFaceTransforms(newVertexCoordinates);
	
	UpdateBoundaryTransforms();
	
	isTensorChanged = true;
}
void Block::setTransform(torch::Tensor &newTransform, optional<torch::Tensor> newFaceTransform) {
	CHECK_INPUT_CUDA(newTransform);
	TORCH_CHECK(newTransform.dim()==velocity.dim(), "new ransform dimensions must match velocity.");
	TORCH_CHECK(newTransform.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform channels must match spatial dimensions");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==newTransform.size(dim-1), "Transform spatial dimension must match velocity's");
	}
	TORCH_CHECK(newTransform.dtype()==velocity.dtype(), "Transform and velocity must have same dtype.");
	
	if(newFaceTransform){
		torch::Tensor faceTransform = newFaceTransform.value();
		CHECK_INPUT_CUDA(faceTransform);
		TORCH_CHECK(faceTransform.dim()==(velocity.dim()+1), "new face transform dimensions must match velocity.");
		TORCH_CHECK(faceTransform.size(0)==1, "batches are not yet supported.");
		TORCH_CHECK(faceTransform.size(1)==getSpatialDims(), "channel dimension must match spatial dimensions.");
		TORCH_CHECK(faceTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform data must match spatial dimensions.");
		for(int dim=2;dim<velocity.dim();++dim){
			TORCH_CHECK(velocity.size(dim)==(faceTransform.size(dim)-1), "Face transform spatial dimension must match velocity's +1.");
		}
		TORCH_CHECK(faceTransform.dtype()==velocity.dtype(), "Face transform and velocity must have same dtype.");
	}
	
	clearCoordsTransforms();
	m_transform = newTransform;
	m_faceTransform = newFaceTransform;
	
	UpdateBoundaryTransforms();
	
	isTensorChanged = true;
}
/*
void Block::setFaceTransform(torch::Tensor &newTransform) {
	TORCH_CHECK(m_transformType==TransformType::TRANSFORM, "cell tranform has to be set before face transform.");
	CHECK_INPUT_CUDA(newTransform);
	TORCH_CHECK(newTransform.dim()==(velocity.dim()+1), "new Transform dimensions must match velocity.");
	TORCH_CHECK(newTransform.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newTransform.size(1)==getSpatialDims(), "channel dimension must match spatial dimensions.");
	TORCH_CHECK(newTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform data must match spatial dimensions.");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==(newTransform.size(dim)-1), "spatial dimension must match +1.");
	}
	TORCH_CHECK(newTransform.dtype()==velocity.dtype(), "Transform and velocity must have same dtype.");
	
	m_faceTransform = newTransform;
	isTensorChanged = true;
}*/
void Block::clearCoordsTransforms(){
	isTensorChanged = isTensorChanged || hasTransform() || hasFaceTransform() || hasVertexCoordinates();
	m_transform = nullopt;
	m_faceTransform = nullopt;
	m_vertexCoordinates = nullopt;
}
torch::Tensor Block::getCellCoordinates() const{
	TORCH_CHECK(hasVertexCoordinates(), "VertexCoordinates are required to compute cell coordinates.")
	switch(getSpatialDims()){
		case 1:
			return torch::avg_pool1d(m_vertexCoordinates.value(), {2}, {1}); // data, kernel size, stride
		case 2:
			return torch::avg_pool2d(m_vertexCoordinates.value(), {2,2}, {1,1});
		case 3:
			return torch::avg_pool3d(m_vertexCoordinates.value(), {2,2,2}, {1,1,1});
		default:
			TORCH_CHECK(false, "only 1-3D is supported.");
			return torch::empty(0);
	}
	
}

torch::Tensor Block::getCellSizes() const{
	if(hasTransform()){
		return torch::unsqueeze(torch::abs(m_transform.value().index({torch::indexing::Ellipsis, -1})), 1);
	} else {
		return torch::ones_like(pressure);
	}
}

index_t Block::GetCoordinateOrientation() const {
	if(!hasTransform()){
		return 1;
	}
	
	torch::Tensor det_sign = torch::sign(m_transform.value().index({torch::indexing::Ellipsis, -1}));
	const index_t sign_min = torch::min(det_sign).cpu().to(torch_kIndex).data_ptr<index_t>()[0];
	const index_t sign_max = torch::max(det_sign).cpu().to(torch_kIndex).data_ptr<index_t>()[0];
	
	if(sign_min!=sign_max){
		return 0;
	}else{
		return sign_min;
	}
}

index_t Block::getSpatialDims() const {
    //return numDims - 2;
	return getParentDomain()->getSpatialDims();
}

index_t Block::getDim(const dim_t dim) const {
    return velocity.size(dim);
}

index_t Block::getAxis(const dim_t dim) const {
	TORCH_CHECK(0<=dim && dim<getSpatialDims(), "Axis index must be within spatial dimensions");
    return velocity.size(velocity.dim()-1 -dim);
}


I4 Block::getSizes() const {
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

I4 Block::getStrides() const {
    const I4 sizes = getSizes();
	return {{.x=1, .y=sizes.x, .z=sizes.x*sizes.y, .w=sizes.x*sizes.y*sizes.z}};
}

index_t Block::ComputeCSRSize() const {
	const I4 size = getSizes();
	const I4 stride = getStrides();
    index_t csrSize = stride.w*(2*getSpatialDims()+1);
	for(dim_t dim = 0; dim<getSpatialDims(); ++dim){
		index_t dimBoundArea = size.a[(dim+1)%3] * size.a[(dim+2)%3];
		if(IsUnconnectedBoundary(dim*2)) csrSize -= dimBoundArea;
		if(IsUnconnectedBoundary(dim*2+1)) csrSize -= dimBoundArea;
	}
    return csrSize;
};

bool Block::IsTensorChanged() const {
	if(isTensorChanged) { return true; }
	for(auto boundary : boundaries){
		if(boundary->IsTensorChanged()){ return true; }
	}
	return false;
}
void Block::setTensorChanged(const bool changed){
	isTensorChanged = changed;
	for(auto boundary : boundaries){
		boundary->setTensorChanged(changed);
	}
}

std::string Block::ToString() const {
	
	std::ostringstream repr;
	I4 size = getSizes();
	I4 stride = getStrides();
	repr << "Block[\"" << name << "\" " << getSpatialDims() << "D";
	repr << ", scalarChannels=" << getPassiveScalarChannels();
	repr << ", transforms=(";
	if(m_vertexCoordinates || m_transform || m_faceTransform){
		if(m_vertexCoordinates) { repr << "C,"; }
		if(m_transform) { repr << "T,"; }
		if(m_faceTransform) { repr << "F,"; }
	} else {
		repr << "none";
	}
	repr << ")";
	//repr << ", dims=" << static_cast<index_t>(numDims) << ", spatial=" << getSpatialDims();
	repr << ", size=" << I4toString(size) << ", stride=" << I4toString(stride);
	repr << ", bounds=( ";
	const index_t numBounds = static_cast<index_t>(boundaries.size());
	for(index_t boundIdx=0; boundIdx<numBounds; ++boundIdx){
		repr << BoundaryIndexToString(boundIdx) << "=" << boundaries[boundIdx]->ToString() << " ";
	}
	repr << ")]";
	return repr.str();
}

// --- Domain ---

