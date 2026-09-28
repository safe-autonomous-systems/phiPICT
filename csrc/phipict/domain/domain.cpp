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
// Domain: blocks, solve data, batched environments and the device atlas.

#include "domain/domain_internal.h"

//Domain::Domain(std::string &name) : name(name), initialized(false) {}
Domain::Domain(const index_t spatialDims, torch::Tensor &viscosity,
		const std::string &name, const torch::Dtype dtype, optional<py::object> pyDtype, const torch::Device device,
		const index_t passiveScalarChannels, optional<torch::Tensor> passiveScalarViscosity)
		: name(name), pyDtype(pyDtype), m_spatialDims(spatialDims), m_passiveScalarChannels(passiveScalarChannels),
		m_dtype(dtype), m_device(device), initialized(false) {
	//
	TORCH_CHECK(0<m_spatialDims && m_spatialDims<4, "spatialDims must be 1, 2, or 3.");
	TORCH_CHECK(m_passiveScalarChannels>=0, "passiveScalarChannels can't be negative.");
	//TORCH_CHECK(m_passiveScalarChannels==1, "Currently only 1 passiveScalarChannel is supported.");

	setViscosity(viscosity);

	/*TORCH_CHECK(!passiveScalarViscosity.has_value(), "passiveScalarViscosity is not yet supported.");
	if(passiveScalarViscosity){
		CHECK_INPUT_CUDA(passiveScalarViscosity.value());
		TORCH_CHECK(passiveScalarViscosity.value().dim()==1, "passiveScalarViscosity must be 1D.");
		TORCH_CHECK(passiveScalarViscosity.value().size(0)==m_passiveScalarChannels, "passiveScalarViscosity size must match passiveScalarChannels.");
	}*/
	if(passiveScalarViscosity){
		setScalarViscosity(passiveScalarViscosity.value());
	}

	TORCH_CHECK(m_device.is_cuda(), "device must be a CUDA device.");
	if(!m_device.has_index()){ m_device.set_index(0); }
	
	valueOptions = torch::TensorOptions().dtype(dtype).layout(torch::kStrided).device(device.type(), device.index());
}
#ifdef DTOR_MSG
Domain::~Domain(){
	py::print("Domain dtor");
	//DetachFwd();
}
#endif

std::shared_ptr<Domain> Domain::Copy(optional<std::string> newName) const{
	std::string n = newName.value_or(name+"_copy");
	torch::Tensor v = viscosity;
	std::shared_ptr<Domain> cDomain = std::make_shared<Domain>(getSpatialDims(), v, n, getDtype(), pyDtype, getDevice(), getPassiveScalarChannels(), passiveScalarViscosity);

	for(auto block : blocks){
		cDomain->AddBlock(block->Copy());
	}

	// copy connected boundaries
	for(index_t blockIdx=0, numBlocks=blocks.size(); blockIdx<numBlocks; ++blockIdx){
		for(index_t boundIdx=0, numBounds=blocks[blockIdx]->boundaries.size(); boundIdx<numBounds; ++boundIdx){
			if(blocks[blockIdx]->boundaries[boundIdx]->type==BoundaryType::CONNECTED_GRID){
				std::shared_ptr<ConnectedBoundary> boundary = std::static_pointer_cast<ConnectedBoundary> (blocks[blockIdx]->boundaries[boundIdx]);
				index_t connectedIdx = getBlockIdx(boundary->getConnectedBlock());
				std::shared_ptr<ConnectedBoundary> cBoundary = std::make_shared<ConnectedBoundary>(cDomain->getBlock(connectedIdx), boundary->axes, shared_from_this());
				cDomain->getBlock(blockIdx)->setBoundary(boundIdx, cBoundary);
			}
		}
	}

	cDomain->m_epotEnabled = m_epotEnabled;
	cDomain->m_epotNonOrthoFlags = m_epotNonOrthoFlags;
	cDomain->m_epotUseFaceTransform = m_epotUseFaceTransform;
	cDomain->m_advectionScheme = m_advectionScheme;

	return cDomain;
}
std::shared_ptr<Domain> Domain::Clone(optional<std::string> newName) const{
	std::string n = newName.value_or(name+"_clone");
	torch::Tensor v = viscosity.clone();
	optional<torch::Tensor> psv = cloneOptionalTensor(passiveScalarViscosity);
	std::shared_ptr<Domain> cDomain = std::make_shared<Domain>(getSpatialDims(), v, n, getDtype(), pyDtype, getDevice(), getPassiveScalarChannels(), psv);

	for(auto block : blocks){
		cDomain->AddBlock(block->Clone());
	}

	// copy connected boundaries
	for(index_t blockIdx=0, numBlocks=blocks.size(); blockIdx<numBlocks; ++blockIdx){
		for(index_t boundIdx=0, numBounds=blocks[blockIdx]->boundaries.size(); boundIdx<numBounds; ++boundIdx){
			if(blocks[blockIdx]->boundaries[boundIdx]->type==BoundaryType::CONNECTED_GRID){
				std::shared_ptr<ConnectedBoundary> boundary = std::static_pointer_cast<ConnectedBoundary> (blocks[blockIdx]->boundaries[boundIdx]);
				index_t connectedIdx = getBlockIdx(boundary->getConnectedBlock());
				std::shared_ptr<ConnectedBoundary> cBoundary = std::make_shared<ConnectedBoundary>(cDomain->getBlock(connectedIdx), boundary->axes, shared_from_this());
				cDomain->getBlock(blockIdx)->setBoundary(boundIdx, cBoundary);
			}
		}
	}

	cDomain->m_epotEnabled = m_epotEnabled;
	cDomain->m_epotNonOrthoFlags = m_epotNonOrthoFlags;
	cDomain->m_epotUseFaceTransform = m_epotUseFaceTransform;
	cDomain->m_advectionScheme = m_advectionScheme;

	return cDomain;
}
std::shared_ptr<Domain> Domain::To(const torch::Dtype dtype, optional<std::string> newName){
	if(dtype==m_dtype){
		return shared_from_this();
	} else {
		TORCH_CHECK(false, "To(dtype) not implemented.");
		return shared_from_this();
	}
}
std::shared_ptr<Domain> Domain::To(const torch::Device device, optional<std::string> newName){
	if(device==getDevice()){
		return shared_from_this();
	} else {
		TORCH_CHECK(device.is_cuda(), "device must be a CUDA device.");
		TORCH_CHECK(false, "To(device) not implemented.");
		return shared_from_this();
	}
}

void Domain::AddBlock(std::shared_ptr<Block> block){
	TORCH_CHECK(block->getSpatialDims()==getSpatialDims(), "Number of block spatial dimensions does not match domain.");
	TORCH_CHECK(block->getPassiveScalarChannels()==getPassiveScalarChannels(), "Number of block passive scalar channels does not match domain.");
	TORCH_CHECK(block->getDtype()==getDtype(), "Data typ does not match.");
	TORCH_CHECK(block->getDevice()==getDevice(), "Device does not match.");

	blocks.push_back(block);
	initialized = false;
}


std::shared_ptr<Block> Domain::CreateBlock( optional<torch::Tensor> velocity, optional<torch::Tensor> pressure, optional<torch::Tensor> passiveScalar,
		optional<torch::Tensor> vertexCoordinates, const std::string &name){
	/*CHECK_INPUT_CUDA(velocity);
	TORCH_CHECK(velocity.dim()==m_spatialDims+2, "velocity must be " + std::to_string(m_spatialDims) + "D.");
	TORCH_CHECK(velocity.size(0)==1, "Batches (dim 0) are not yet supported (must be 1).");
	TORCH_CHECK(velocity.size(1)==m_spatialDims, "Channels (dim 1) of velocity must match spatial dimensions ()" + std::to_string(m_spatialDims) + ").");
	TORCH_CHECK(velocity.dtype()==m_dtype, "velocity has wrong dtype.");
	TORCH_CHECK(velocity.device()==m_device, "velocity has wrong device.");

	TORCH_CHECK(passiveScalar.has_value()==hasPassiveScalar(), "");
	if(passiveScalar){
		TORCH_CHECK(passiveScalar.value().size(1)==m_passiveScalarChannels, "passiveScalar channels do not match domain.");
	}*/
	
	std::shared_ptr<Block> p_block = std::make_shared<Block>(velocity, pressure, passiveScalar, vertexCoordinates, name, shared_from_this());
	
	TORCH_CHECK(p_block->getDevice()==getDevice(), "Block has wrong device");
	//TORCH_CHECK(CompareDevice(p_block->getDevice(), getDevice()), "Block has wrong device");
	//TORCH_CHECK(p_block->getDevice().type()==getDevice().type(), "Block has wrong device type");
	//TORCH_CHECK(p_block->getDevice().index()==getDevice().index(), "Block has wrong device index: Block has "
	//	+ std::to_string(p_block->getDevice().index()) + ", domain expects " + std::to_string(getDevice().index()));
	//TORCH_CHECK(p_block->getDtype()==getDtype(), "Block has wrong dtype");
	//TORCH_CHECK(p_block->getSpatialDims()==getSpatialDims(), "Block must be " + std::to_string(m_spatialDims) + "D.");
	//TORCH_CHECK(p_block->getPassiveScalarChannels()==getPassiveScalarChannels(), "passiveScalar channels do not match domain.");
	
	AddBlock(p_block);
	
	return p_block;
}

std::shared_ptr<Block> Domain::CreateBlockWithSize(I4 size, std::string &name){
	TORCH_CHECK(size.x>2, "size.x must be at least 3.");
	if(getSpatialDims()>1) TORCH_CHECK(size.y>2, "size.y must be at least 3.");
	if(getSpatialDims()>2) TORCH_CHECK(size.z>2, "size.z must be at least 3.");
	size.w = getSpatialDims();

	std::shared_ptr<Block> p_block = std::make_shared<Block>(size, name, shared_from_this());
	if(hasPassiveScalar()){
		p_block->CreatePassiveScalar();
	}
	
	AddBlock(p_block);

	return p_block;
}

/*
std::shared_ptr<Block> Domain::CreateBlockFromCoords(torch::Tensor vertexCoordinates, const std::string &name){
	CHECK_INPUT_CUDA(vertexCoordinates);
	TORCH_CHECK(vertexCoordinates.dim()==m_spatialDims+2, "vertexCoordinates must be " + std::to_string(m_spatialDims) + "D.");
	TORCH_CHECK(vertexCoordinates.size(0)==1, "Batches (dim 0) are not yet supported (must be 1).");
	TORCH_CHECK(vertexCoordinates.size(1)==m_spatialDims, "Channels (dim 1) of vertexCoordinates must match spatial dimensions ()" + std::to_string(m_spatialDims) + ").");
	TORCH_CHECK(vertexCoordinates.dtype()==m_dtype, "vertexCoordinates has wrong dtype.");
	TORCH_CHECK(vertexCoordinates.device()==m_device, "vertexCoordinates has wrong device.");
	
	I4 size = {.a={0}};
	size.w = m_spatialDims;
	for(index_t dim=0; dim<m_spatialDims; ++dim){
		size.a[dim] = vertexCoordinates.size(m_spatialDims + 1 - dim) - 1;
	}

	std::shared_ptr<Block> p_block = CreateBlockWithSize(size, name);
	
	p_block->setVertexCoordinates(vertexCoordinates);
	
	AddBlock(p_block);

	return p_block;
}*/

torch::Tensor Domain::getMaxVelocity(const bool withBounds, const bool computational) const{
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	torch::Tensor maxVel = blocks[0]->getMaxVelocity(withBounds, computational);
	for(size_t blockIdx = 1; blockIdx < blocks.size(); ++blockIdx){
		maxVel = torch::maximum(maxVel, blocks[blockIdx]->getMaxVelocity(withBounds, computational));
	}
	
	return maxVel;
}

torch::Tensor Domain::getMaxVelocityPerEnv(const bool withBounds, const bool computational) const{
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	torch::Tensor maxVel = blocks[0]->getMaxVelocityPerEnv(withBounds, computational);
	for(size_t blockIdx = 1; blockIdx < blocks.size(); ++blockIdx){
		maxVel = torch::maximum(maxVel, blocks[blockIdx]->getMaxVelocityPerEnv(withBounds, computational));
	}
	
	return maxVel;
}

torch::Tensor Domain::getMaxVelocityMagnitude(const bool withBounds, const bool computational) const{
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	torch::Tensor maxMag = blocks[0]->getMaxVelocityMagnitude(withBounds, computational);
	for(size_t blockIdx = 1; blockIdx < blocks.size(); ++blockIdx){
		maxMag = torch::maximum(maxMag, blocks[blockIdx]->getMaxVelocityMagnitude(withBounds, computational));
	}
	
	return maxMag;
}
bool Domain::hasVertexCoordinates() const {
	for(auto block : blocks){
		if(!block->hasVertexCoordinates()){
			return false;
		}
	}
	
	return getNumBlocks()>0;
}
std::vector<torch::Tensor> Domain::getVertexCoordinates() const {
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	std::vector<torch::Tensor> coordList;
	for(auto block : blocks){
		TORCH_CHECK(block->hasVertexCoordinates(), "a block is missing vertex coordinates");
		coordList.push_back(block->m_vertexCoordinates.value());
	}
	
	return coordList;
}

index_t Domain::GetCoordinateOrientation() const {
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	index_t sign = blocks[0]->GetCoordinateOrientation();
	for(index_t i=1; i<getNumBlocks(); ++i){
		if(blocks[i]->GetCoordinateOrientation() != sign){
			return 0;
		}
	}
	
	return sign;
}

bool Domain::hasPrescribedBoundary() const {
	for(const auto &block : blocks){
		if(block->hasPrescribedBoundary()){
			return true;
		}
	}
	return false;
}

bool Domain::isAllFixedBoundariesPassiveScalarTypeStatic() const{
	for(const auto &block : blocks){
		if(!block->isAllFixedBoundariesPassiveScalarTypeStatic()){
			return false;
		}
	}
	return true;
}

torch::Tensor Domain::GetGlobalFluxBalance() const {
	torch::Tensor fluxSum = torch::zeros({1}, valueOptions);
	torch::Tensor mOne = torch::tensor({-1}, valueOptions);
	
	for(auto block : blocks){
		// std::shared_ptr<Block> block;
		index_t boundIdx = 0;
		for(auto boundary : block->getBoundaries()){
			switch(boundary->type){
				case BoundaryType::FIXED: {
					std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary>(boundary);
					// per environment: sum over everything but the batch dimension
					torch::Tensor fluxes = bound->GetFluxes();
					torch::Tensor boundaryFlux = fluxes.reshape({fluxes.size(0), -1}).sum(1);
					
					if(!(boundIdx&1)){ // is lower boundary
						boundaryFlux = boundaryFlux * mOne;
					}
					
					fluxSum = fluxSum + boundaryFlux;
					break;
				}
				case BoundaryType::DIRICHLET:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::NEUMANN:{
					TORCH_CHECK(false, "Old boundaries not supported.");
					break;
				}
				default:
					break;
			}
			++boundIdx;
		}
	}
	
	return fluxSum;
}
bool Domain::CheckGlobalFluxBalance(const double eps) const {
	// largest imbalance over the batched environments
	const double fluxBalance = GetGlobalFluxBalance().abs().max().cpu().to(torch::kFloat64).item<double>();
	return fluxBalance<eps;
}

void Domain::setViscosity(torch::Tensor &viscosity){
	CHECK_INPUT_HOST(viscosity);
	TORCH_CHECK(viscosity.dim()==1, "viscosity must be 1D.");
	TORCH_CHECK(viscosity.size(0)==1, "viscosity must be a scalar.");
	
	this->viscosity = viscosity;
	if(initialized){
		AT_DISPATCH_FLOATING_TYPES(getDtype(), "SetupDomainGPU", ([&] {
			SetViscosityGPU<scalar_t>();
		}));
	}
}

bool Domain::hasBlockViscosity() const {
	for(const auto &block : blocks){
		if(block->hasViscosity()){
			return true;
		}
	}
	return false;
}

template <typename scalar_t>
void Domain::SetViscosityGPU(){
	for(index_t b=0; b<getBatchSize(); ++b){
		DomainGPU<scalar_t> *p_host_domain= reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_host) + b;
		DomainGPU<scalar_t> *p_device_domain= reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_device) + b;
		p_host_domain->viscosity = viscosity.data_ptr<scalar_t>()[0];
		CopyToGPU(&p_device_domain->viscosity, &p_host_domain->viscosity, sizeof(scalar_t));
	}
}
void Domain::setTimeStep(const torch::Tensor &timeStep){
	TORCH_CHECK(initialized, "Domain is not initialized. Run domain.PrepareSolve() after setup.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1 || timeStep.size(0)==getBatchSize(),
		"timeStep must have size 1 (shared) or the batch size " + std::to_string(getBatchSize()) + ".");
	TORCH_CHECK(timeStep.scalar_type()==getDtype(), "Data type of timeStep does not match.");
	AT_DISPATCH_FLOATING_TYPES(getDtype(), "SetTimeStepGPU", ([&] {
		SetTimeStepGPU<scalar_t>(timeStep);
	}));
}

template <typename scalar_t>
void Domain::SetTimeStepGPU(const torch::Tensor &timeStep){
	const index_t B = getBatchSize();
	const torch::Tensor ts = timeStep.contiguous();
	const scalar_t *p_ts = ts.data_ptr<scalar_t>();
	DomainGPU<scalar_t> *p_host_domains = reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_host);
	std::vector<scalar_t> values(B);
	bool changed = false;
	for(index_t b=0; b<B; ++b){
		values[b] = p_ts[ts.size(0)==1 ? 0 : b];
		changed |= p_host_domains[b].timeStep!=values[b];
		p_host_domains[b].timeStep = values[b];
	}
	if(changed){
		// only the timeStep field of every environment's domain: the host copy holds host pointers
		DomainGPU<scalar_t> *p_device_domains = reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_device);
		CopyToGPUStrided(&p_device_domains[0].timeStep, sizeof(DomainGPU<scalar_t>), values.data(), sizeof(scalar_t), B);
	}
}

void Domain::setScalarViscosity(torch::Tensor &viscosity){
	TORCH_CHECK(viscosity.dim()==1, "Scalar viscosity must be 1D.");
	TORCH_CHECK(viscosity.size(0)==1 || viscosity.size(0)==m_passiveScalarChannels, "Scalar viscosity must be static (a scalar) or match passive scalar channels.");
	TORCH_CHECK(viscosity.scalar_type()==getDtype(), "Data type of scalar viscosity does not match.");
	CHECK_INPUT_CUDA(viscosity);
	//TORCH_CHECK(viscosity.device()==getDevice(), "Device of scalar viscosity does not match.");

	passiveScalarViscosity = viscosity;
	isTensorChanged=true;
}
void Domain::clearScalarViscosity() {
	
	passiveScalarViscosity = nullopt;
	isTensorChanged=true;
}
bool Domain::hasPassiveScalarBlockViscosity() const {
	for(const auto &block : blocks){
		if(block->hasPassiveScalarViscosity()){
			return true;
		}
	}
	return false;
}

void Domain::PrepareSolve(){
	//if(initialized) return;
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	TORCH_CHECK(viscosity.scalar_type()==getDtype(), "Data type of viscosity does not match.");
	
	//const bool hasPassiveScalar = blocks[0]->hasPassiveScalar();
	//const index_t passiveScalarChannels = blocks[0]->getPassiveScalarChannels();
	
	index_t csrSize = 0;
	totalSize = 0;
	for(auto block : blocks){
		block->csrOffset = csrSize;
		block->globalOffset = totalSize;
		
		csrSize += block->ComputeCSRSize();
		totalSize += block->getStrides().w;
		
		
		//for(const auto boundary : block->boundaries){
		const index_t numBounds = static_cast<index_t>(block->boundaries.size());
		for(index_t boundIdx=0; boundIdx<numBounds; ++boundIdx){
			std::shared_ptr<Boundary> boundary = block->boundaries[boundIdx];
			switch (boundary->type)
			{
			case BoundaryType::DIRICHLET:
			case BoundaryType::DIRICHLET_VARYING: {
				py::print("Warning: Dirichlet(Varying/Static) boundaries are deprecated, use FixedBoundary instead.");
			}
			/*case BoundaryType::FIXED: {
				std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
				break;
			}*/
			case BoundaryType::CONNECTED_GRID: {
				//std::shared_ptr<ConnectedBoundary> bound = std::dynamic_pointer_cast<ConnectedBoundary> (boundary);
				std::shared_ptr<ConnectedBoundary> bound = std::static_pointer_cast<ConnectedBoundary> (boundary);
				std::shared_ptr<Block> otherBlock = bound->getConnectedBlock();
				TORCH_CHECK(getBlockIdx(otherBlock)>=0, "Connected block is not part of this domain.");
				//check that the connection goes both ways
				const index_t otherBoundIdx = bound->axes[0];
				std::shared_ptr<Boundary> otherBoundary = otherBlock->getBoundary(otherBoundIdx);
				TORCH_CHECK(otherBoundary->type==BoundaryType::CONNECTED_GRID, "Mismatch in block connection: connected block is not connected at the target face.");
				std::shared_ptr<ConnectedBoundary> otherBound = std::static_pointer_cast<ConnectedBoundary> (otherBoundary);
				TORCH_CHECK(block==otherBound->getConnectedBlock(), "Mismatch in block connection: connected block is connected to a different block at the target face.");
				TORCH_CHECK(boundIdx==otherBound->axes[0], "Mismatch in block connection: connected block is connected to a different face of the source block.");
				//TODO: check axis alignments
				// axis sizes are check on adding the boundary to a block
				const dim_t spatialDims = getSpatialDims();
				if(spatialDims==2){
					TORCH_CHECK(bound->getConnectionAxisDirection(1)==otherBound->getConnectionAxisDirection(1), "Direction of connection does not match.");
				}
				if(spatialDims==3){
					//valid configs:
					if(bound->getConnectionAxis(1) == (bound->getConnectedFace() + 1)%spatialDims){
						// axis order not swapped
						TORCH_CHECK(bound->getConnectionAxis(2) == (bound->getConnectedFace() + 2)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(1) == (otherBound->getConnectedFace() + 1)%spatialDims, "Invalid connection configuration: axis1 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(2) == (otherBound->getConnectedFace() + 2)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(bound->getConnectionAxisDirection(1) == otherBound->getConnectionAxisDirection(1), "Invalid connection configuration: Direction of connection axis1-axis1 does not match.");
						TORCH_CHECK(bound->getConnectionAxisDirection(2) == otherBound->getConnectionAxisDirection(2), "Invalid connection configuration: Direction of connection axis2-axis2 does not match.");
						
					} else if(bound->getConnectionAxis(1) == (bound->getConnectedFace() + 2)%spatialDims){
						// axis order swapped
						TORCH_CHECK(bound->getConnectionAxis(2) == (bound->getConnectedFace() + 1)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(1) == (otherBound->getConnectedFace() + 2)%spatialDims, "Invalid connection configuration: axis1 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(2) == (otherBound->getConnectedFace() + 1)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(bound->getConnectionAxisDirection(1) == otherBound->getConnectionAxisDirection(2), "Invalid connection configuration: Direction of connection axis1-axis2 does not match.");
						TORCH_CHECK(bound->getConnectionAxisDirection(2) == otherBound->getConnectionAxisDirection(1), "Invalid connection configuration: Direction of connection axis2-axis1 does not match.");
						
					} else {
						TORCH_CHECK(false, "Invalid connection configuration: axis1 has invalid target axis.");
					}
				}
				break;
			}
			case BoundaryType::PERIODIC: {
				TORCH_CHECK(block->boundaries[boundIdx^1]->type==BoundaryType::PERIODIC, "Opposite boundary must also be periodic.");
				break;
			}
			default:
				break;
			}
		}
	}

	//valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	//auto indexOptions = torch::TensorOptions().dtype(torch_kIndex).layout(torch::kStrided).device(getDevice().type(), getDevice().index());

	const index_t batchSize = getBatchSize();
	for(auto block : blocks){
		TORCH_CHECK(block->getBatchSize()==batchSize, "All blocks must have the same batch size.");
	}
	C = std::make_shared<CSRmatrix>(csrSize, totalSize, getDtype(), getDevice());
	P = std::make_shared<CSRmatrix>(csrSize, totalSize, getDtype(), getDevice());
	if(batchSize>1){
		// batched environments: one matrix per environment, same (shared) sparsity pattern
		C->value = torch::zeros(int64_t(csrSize)*batchSize, valueOptions);
		P->value = torch::zeros(int64_t(csrSize)*batchSize, valueOptions);
	}
	//A = torch::zeros(getBatchSize()*totalSize, valueOptions);
	CreateA();
#ifdef WITH_GRAD
	C_grad = C->WithZeroValue(); //shared indexing tensors
	P_grad = P->WithZeroValue(); //shared indexing tensors
#endif
	
	if(hasPassiveScalar()){
		//scalarRHS = torch::zeros(getBatchSize()*totalSize, valueOptions);
		CreateScalarRHS();
		//scalarResult = torch::zeros(getBatchSize()*totalSize, valueOptions);
		CreateScalarResult();
	}
	
	//velocityRHS = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	CreateVelocityRHS();
	//velocityResult = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	CreateVelocityResult();
	
	//pressureRHS = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	CreatePressureRHS();
	//pressureRHSdiv = torch::zeros(getBatchSize()*totalSize, valueOptions);
	CreatePressureRHSdiv();
	//pressureResult = torch::zeros(getBatchSize()*totalSize, valueOptions);
	CreatePressureResult();
	isTensorChanged=true;
	
	
	AT_DISPATCH_FLOATING_TYPES(getDtype(), "SetupDomainGPU", ([&] {
		SetupDomainGPU<scalar_t>();
	}));

	initialized = true;

	if (m_epotEnabled) {
		SetupEpotOnDomain(m_epotNonOrthoFlags, m_epotUseFaceTransform);
	}
}

void Domain::DetachFwd() {
	for(std::shared_ptr<Block> block : blocks){
		block->DetachFwd();
	}
	A = A.detach();
	C->Detach();
	P->Detach();
	if(!IsTensorEmpty(scalarRHS)){ scalarRHS = scalarRHS.detach(); }
	if(!IsTensorEmpty(scalarResult)){ scalarResult = scalarResult.detach(); }
	velocityRHS = velocityRHS.detach();
	velocityResult = velocityResult.detach();
	pressureRHS = pressureRHS.detach();
	pressureRHSdiv = pressureRHSdiv.detach();
	pressureResult = pressureResult.detach();
	if(Epot) { (*Epot)->Detach(); }
	if(epotRHS) { epotRHS = epotRHS.value().detach(); }
	if(epotResult) { epotResult = epotResult.value().detach(); }
}
void Domain::DetachGrad() {
	for(std::shared_ptr<Block> block : blocks){
		block->DetachGrad();
	}
	A_grad = A_grad.detach();
	C_grad->Detach();
	P_grad->Detach();
	//P_grad->Detach();
	if(!IsTensorEmpty(scalarRHS_grad)){ scalarRHS_grad = scalarRHS_grad.detach(); }
	if(!IsTensorEmpty(scalarResult_grad)){ scalarResult_grad = scalarResult_grad.detach(); }
	velocityRHS_grad = velocityRHS_grad.detach();
	velocityResult_grad = velocityResult_grad.detach();
	pressureRHS_grad = pressureRHS_grad.detach();
	pressureRHSdiv_grad = pressureRHSdiv_grad.detach();
	pressureResult_grad = pressureResult_grad.detach();
	if(hasEpotResultGrad()) { epotResult_grad = epotResult_grad.value().detach(); }
}
void Domain::Detach() {
	DetachFwd();
	DetachGrad();
}
void Domain::CreatePressureOnBlocks(){
	for(auto block : blocks){
		block->CreatePressure();
	}
}
void Domain::CreateEpotOnBlocks(){
	for(auto block : blocks){
		block->CreateEpot();
	}
}
void Domain::CreateVelocityOnBlocks(){
	for(auto block : blocks){
		block->CreateVelocity();
	}
}

void Domain::CreatePassiveScalarOnBlocks(){
	if(hasPassiveScalar()){
		for(auto block : blocks){
			block->CreatePassiveScalar();
		}
	}
}
/* void Domain::clearPassiveScalarOnBlocks(){
	for(auto block : blocks){
		block->clearPassiveScalar();
	}
} */
bool Domain::hasPassiveScalar() const{
	return m_passiveScalarChannels > 0;
}
index_t Domain::getPassiveScalarChannels() const{
	return m_passiveScalarChannels;
}

bool Domain::CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const std::string &name){
	CHECK_INPUT_CUDA(tensor);
	TORCH_CHECK(tensor.dim()==1, "Dimensions of " + name + " must be 1 (flat).");
	TORCH_CHECK(tensor.size(0)==totalSize*channels*getBatchSize(), "Size of " + name + " must be " + std::to_string(totalSize*channels*getBatchSize()) + ".");
	TORCH_CHECK(tensor.scalar_type()==getDtype(), name + " has wrong dtype.");
	return true;
}
void Domain::setA(torch::Tensor &a){
	CheckDataTensor(a, 1, "A");
	A = a;
	isTensorChanged=true;
}
void Domain::CreateA(){
	A = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}

void Domain::setScalarRHS(torch::Tensor &srhs){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(srhs, getPassiveScalarChannels(), "ScalarRHS");
	scalarRHS = srhs;
	isTensorChanged=true;
}
void Domain::CreateScalarRHS(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarRHS = torch::zeros(getBatchSize()*totalSize * getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}
void Domain::setScalarResult(torch::Tensor &sr){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(sr, getPassiveScalarChannels(), "ScalarResult");
	scalarResult = sr;
	isTensorChanged=true;
}
void Domain::CreateScalarResult(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarResult = torch::zeros(getBatchSize()*totalSize * getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}
void Domain::setVelocityRHS(torch::Tensor &vrhs){
	CheckDataTensor(vrhs, getSpatialDims(), "VelocityRHS");
	velocityRHS = vrhs;
	isTensorChanged=true;
}
void Domain::CreateVelocityRHS(){
	velocityRHS = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setVelocityResult(torch::Tensor &vr){
	CheckDataTensor(vr, getSpatialDims(), "VelocityResult");
	velocityResult = vr;
	isTensorChanged=true;
}
void Domain::CreateVelocityResult(){
	velocityResult = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureRHS(torch::Tensor &prhs){
	CheckDataTensor(prhs, getSpatialDims(), "PressureRHS");
	pressureRHS = prhs;
	isTensorChanged=true;
}
void Domain::CreatePressureRHS(){
	pressureRHS = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureRHSdiv(torch::Tensor &prhsd){
	CheckDataTensor(prhsd, 1, "PressureRHSdiv");
	pressureRHSdiv = prhsd;
	isTensorChanged=true;
}
void Domain::CreatePressureRHSdiv(){
	pressureRHSdiv = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureResult(torch::Tensor &pr){
	CheckDataTensor(pr, 1, "pressureResult");
	pressureResult = pr;
	isTensorChanged=true;
}
void Domain::CreatePressureResult(){
	pressureResult = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}

bool Domain::hasEpot() const {
	return epotRHS.has_value() && epotResult.has_value() && Epot.has_value();
}
void Domain::setEpotRHS(torch::Tensor &erhs){
	CheckDataTensor(erhs, 1, "EpotRHS");
	epotRHS = erhs;
	isTensorChanged=true;
}
void Domain::CreateEpotRHS(){
	epotRHS = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}
void Domain::setEpotResult(torch::Tensor &er){
	CheckDataTensor(er, 1, "EpotResult");
	epotResult = er;
	isTensorChanged=true;
}
void Domain::CreateEpotResult(){
	epotResult = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}
void Domain::CreateEpotMatrix(){
	TORCH_CHECK(initialized, "Domain is not initialized. Run domain.PrepareSolve() before CreateEpotMatrix().");
	// constant matrix: shared by all batched environments
	Epot = std::make_shared<CSRmatrix>(P->getNnz(), P->getRows(), getDtype(), getDevice());
	isTensorChanged=true;
}

void Domain::setAdvectionScheme(AdvectionScheme scheme){
	m_advectionScheme = scheme;
	// mirrored into DomainGPU by UpdateDomain()
	isTensorChanged = true;
}

void Domain::SetupEpotOnDomain(int8_t nonOrthoFlags, bool useFaceTransform){
	m_epotNonOrthoFlags = nonOrthoFlags;
	m_epotUseFaceTransform = useFaceTransform;
	m_epotEnabled = true;
	if(!epotRHS){ CreateEpotRHS(); }
	const bool hadEpotResult = epotResult.has_value();
	if(!hadEpotResult){ CreateEpotResult(); }
	CreateEpotMatrix();
	UpdateDomain();
	SetupEpotMatrix(shared_from_this(), nonOrthoFlags, useFaceTransform);
	bool allBlocksHadEpot = true;
	for(auto block : blocks){
		if(!block->hasEpot()){
			block->CreateEpot();
			allBlocksHadEpot = false;
		}
	}
	UpdateDomain();
	if(!hadEpotResult && allBlocksHadEpot){
		CopyEpotResultFromBlocks(shared_from_this());
	}
}

// --- Thin-wall epot system ---

#ifdef WITH_GRAD

void Domain::CreatePassiveScalarGradOnBlocks(){
	for(auto block : blocks){
		block->CreatePassiveScalarGrad();
	}
}
void Domain::CreateVelocityGradOnBlocks(){
	for(auto block : blocks){
		block->CreateVelocityGrad();
	}
}
void Domain::CreateVelocitySourceGradOnBlocks(){
	for(auto block : blocks){
		block->CreateVelocitySourceGrad();
	}
}
void Domain::CreatePressureGradOnBlocks(){
	for(auto block : blocks){
		block->CreatePressureGrad();
	}
}

void Domain::CreatePassiveScalarGradOnBoundaries(){
	for(auto block : blocks){
		block->CreatePassiveScalarGradOnBoundaries();
	}
}
void Domain::CreateVelocityGradOnBoundaries(){
	for(auto block : blocks){
		block->CreateVelocityGradOnBoundaries();
	}
}
void Domain::setViscosityGrad(torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.dim()==1 && t.size(0)==1, "velocity grad must have shape (1).")
	TORCH_CHECK(t.scalar_type()==getDtype(), "velocity grad has wrong dtype.");
	
	viscosity_grad = t;
	isTensorChanged=true;
}
void Domain::clearViscosityGrad(){
	if(hasViscosityGrad()){
		viscosity_grad = nullopt;
		isTensorChanged = true;
	}
}
void Domain::CreateViscosityGrad(){
	torch::Tensor v_grad = torch::zeros({1}, valueOptions);
	setViscosityGrad(v_grad);
	for(auto block : blocks){
		block->CreateViscosityGrad();
	}
	CreatePassiveScalarViscosityGrad();
}

void Domain::setPassiveScalarViscosityGrad(torch::Tensor &t){
	TORCH_CHECK(hasPassiveScalarViscosity(), "Domain has no passive scalar viscosity.");
	TORCH_CHECK(t.dim()==1, "Scalar viscosity grad must be 1D.");
	TORCH_CHECK(t.size(0)==passiveScalarViscosity.value().size(0), "Scalar viscosity grad shape must match Scalar viscosity.");
	TORCH_CHECK(t.scalar_type()==getDtype(), "Data type of scalar viscosity does not match.");
	CHECK_INPUT_CUDA(t);
	
	passiveScalarViscosity_grad = t;
	isTensorChanged = true;
}
void Domain::clearPassiveScalarViscosityGrad(){
	if(hasPassiveScalarViscosityGrad()){
		passiveScalarViscosity_grad = nullopt;
		isTensorChanged = true;
	}
}
void Domain::CreatePassiveScalarViscosityGrad(){
	if(hasPassiveScalarViscosity()){
		torch::Tensor v = torch::zeros_like(passiveScalarViscosity.value());
		setPassiveScalarViscosityGrad(v);
	} else {
		clearPassiveScalarViscosityGrad();
	}
}

void Domain::setAGrad(torch::Tensor &tensor){
	CheckDataTensor(tensor, 1, "AGrad");
	A_grad = tensor;
	isTensorChanged=true;
}
void Domain::CreateAGrad(){
	A_grad = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}

void Domain::setScalarRHSGrad(torch::Tensor &srhsg){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(srhsg, getPassiveScalarChannels(), "ScalarRHSGrad");
	scalarRHS_grad = srhsg;
	isTensorChanged=true;
}
void Domain::CreateScalarRHSGrad(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarRHS_grad = torch::zeros(getBatchSize()*totalSize*getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}
void Domain::setScalarResultGrad(torch::Tensor &srg){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(srg, getPassiveScalarChannels(), "ScalarResultGrad");
	scalarResult_grad = srg;
	isTensorChanged=true;
}
void Domain::CreateScalarResultGrad(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarResult_grad = torch::zeros(getBatchSize()*totalSize*getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}

void Domain::setVelocityRHSGrad(torch::Tensor &vrhs){
	CheckDataTensor(vrhs, getSpatialDims(), "velocityRHS_grad");
	velocityRHS_grad = vrhs;
	isTensorChanged=true;
}
void Domain::CreateVelocityRHSGrad(){
	velocityRHS_grad = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setVelocityResultGrad(torch::Tensor &vr){
	CheckDataTensor(vr, getSpatialDims(), "velocityResult_grad");
	velocityResult_grad = vr;
	isTensorChanged=true;
}
void Domain::CreateVelocityResultGrad(){
	velocityResult_grad = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}

void Domain::setPressureRHSGrad(torch::Tensor &prhs){
	CheckDataTensor(prhs, getSpatialDims(), "pressureRHS_grad");
	pressureRHS_grad = prhs;
	isTensorChanged=true;
}
void Domain::CreatePressureRHSGrad(){
	pressureRHS_grad = torch::zeros(getBatchSize()*totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureRHSdivGrad(torch::Tensor &prhsd){
	CheckDataTensor(prhsd, 1, "pressureRHSdiv_grad");
	pressureRHSdiv_grad = prhsd;
	isTensorChanged=true;
}
void Domain::CreatePressureRHSdivGrad(){
	pressureRHSdiv_grad = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureResultGrad(torch::Tensor &pr){
	CheckDataTensor(pr, 1, "pressureResult_grad");
	pressureResult_grad = pr;
	isTensorChanged=true;
}
void Domain::CreatePressureResultGrad(){
	pressureResult_grad = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}

void Domain::setEpotResultGrad(torch::Tensor &erg){
	CheckDataTensor(erg, 1, "epotResult_grad");
	epotResult_grad = erg;
	isTensorChanged=true;
}
void Domain::CreateEpotResultGrad(){
	epotResult_grad = torch::zeros(getBatchSize()*totalSize, valueOptions);
	isTensorChanged=true;
}
void Domain::CreateEpotGradOnBlocks(){
	for(auto block : blocks) block->CreateEpotGrad();
}

#endif //WITH_GRAD

template <typename scalar_t>
void Domain::SetupDomainGPU(){
	
	const size_t domainAlignment = alignof(DomainGPU<scalar_t>); //std::alignment_of<DomainGPU<scalar_t>>::value;
	const size_t blockAlignment =  alignof(BlockGPU<scalar_t>); //std::alignment_of<BlockGPU<scalar_t>>::value;
	const index_t numBlocks = blocks.size();
	const index_t batchSize = getBatchSize();
	// Atlas layout for B batched environments: [DomainGPU x B][BlockGPU x numBlocks x B].
	// Kernels select their environment's domain with p_domain[blockIdx.y]; the blocks
	// pointer of domain b points to its own block array.
	const size_t blocksStartOffsetBytes = sizeof(DomainGPU<scalar_t>) * batchSize;
	if(domainAlignment<blockAlignment && blocksStartOffsetBytes%blockAlignment!=0){
		//TODO correct block start offset for alignment
		TORCH_CHECK(false, "Alignment issues.")
	}
	const size_t atlasSizeBytes = blocksStartOffsetBytes + sizeof(BlockGPU<scalar_t>) * numBlocks * batchSize;
	size_t allocSizeBytes = domainAlignment + atlasSizeBytes;
	
	auto byteOptions = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	auto byteOptionsCPU = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided); //.device(getDevice().type(), getDevice().index());
	
	domainCPU = torch::zeros(allocSizeBytes, byteOptionsCPU);
	domainGPU = torch::zeros(allocSizeBytes, byteOptions);
	
	//DomainAtlasSet atlas;
	memset(&atlas, 0, sizeof(DomainAtlasSet));
	atlas.sizeBytes = atlasSizeBytes;
	atlas.blocksOffsetBytes = blocksStartOffsetBytes;
	atlas.p_host = reinterpret_cast<void*>(domainCPU.data_ptr<uint8_t>());
	atlas.p_device = reinterpret_cast<void*>(domainGPU.data_ptr<uint8_t>());
	TORCH_CHECK(std::align(domainAlignment, atlasSizeBytes, atlas.p_host, allocSizeBytes), "Failed to align CPU domain.")
	TORCH_CHECK(std::align(domainAlignment, atlasSizeBytes, atlas.p_device, allocSizeBytes), "Failed to align GPU domain.")
	
	UpdateDomainGPU<scalar_t>();
}

void Domain::UpdateDomain(){
	if(!initialized) { TORCH_CHECK(false, "Domain is not initialized. Run domain.PrepareSolve() after setup."); }
	if(IsTensorChanged()){
		AT_DISPATCH_FLOATING_TYPES(getDtype(), "UpdateDomainGPU", ([&] {
			UpdateDomainGPU<scalar_t>();
		}));
	}
}

/* Pointers of batched environment b. A tensor whose dim 0 is the batch size B holds one slice
 * per environment; with dim 0 of size 1 it is shared by all environments. */
template <typename T>
static T* batchSlicePtr(const torch::Tensor &t, const index_t b, const index_t B, const std::string &name){
	if(!t.defined()) return nullptr;
	T *p = t.data_ptr<T>();
	if(B<=1 || t.dim()==0) return p;
	if(t.size(0)==B) return p + b*t.stride(0);
	TORCH_CHECK(t.size(0)==1, "Batch size (dim 0) of " + name + " must be 1 (shared) or the batch size " + std::to_string(B) + ".");
	return p;
}
template <typename T>
static T* batchSlicePtr(const optional<torch::Tensor> &t, const index_t b, const index_t B, const std::string &name){
	return t.has_value() ? batchSlicePtr<T>(t.value(), b, B, name) : nullptr;
}
/* Gradient tensors of environment b: nullptr if the gradient is not allocated. A gradient of
 * batch size 1 belongs to data shared by all environments, which the gradient kernels
 * accumulate into atomically, so it receives the sum over the environments. */
template <typename T>
static T* batchGradSlicePtr(const torch::Tensor &t, const index_t b, const index_t B, const std::string &name){
	return IsTensorEmpty(t) ? nullptr : batchSlicePtr<T>(t, b, B, name);
}
template <typename T>
static T* batchGradSlicePtr(const optional<torch::Tensor> &t, const index_t b, const index_t B, const std::string &name){
	return t.has_value() ? batchGradSlicePtr<T>(t.value(), b, B, name) : nullptr;
}
/* Flat vectors of the linear systems hold B consecutive environment slices. */
template <typename T>
static T* batchFlatPtr(const torch::Tensor &t, const index_t b, const index_t B){
	if(!t.defined()) return nullptr;
	T *p = t.data_ptr<T>();
	if(B<=1) return p;
	TORCH_CHECK(t.numel()%B==0, "Flat vector size is not a multiple of the batch size.");
	return p + b*(t.numel()/B);
}
template <typename T>
static T* batchFlatPtr(const optional<torch::Tensor> &t, const index_t b, const index_t B){
	return t.has_value() ? batchFlatPtr<T>(t.value(), b, B) : nullptr;
}
template <typename T>
static T* batchFlatGradPtr(const torch::Tensor &t, const index_t b, const index_t B){
	return IsTensorEmpty(t) ? nullptr : batchFlatPtr<T>(t, b, B);
}
/* CSR values: one matrix per environment (same pattern), or one shared matrix. */
template <typename T>
static T* batchCSRValuePtr(const std::shared_ptr<CSRmatrix> &m, const index_t b){
	T *p = m->value.data_ptr<T>();
	return m->getBatchSize()>1 ? p + b*m->getNnz() : p;
}

template <typename scalar_t>
void Domain::UpdateDomainGPU(){
	const index_t B = getBatchSize();
	const index_t numBlocks = blocks.size();
	DomainGPU<scalar_t> *p_domainsCPU = reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_host);
	BlockGPU<scalar_t> *p_blocksCPUall = reinterpret_cast<BlockGPU<scalar_t>*>(reinterpret_cast<uint8_t*>(atlas.p_host) + atlas.blocksOffsetBytes);
	BlockGPU<scalar_t> *p_blocksGPUall = reinterpret_cast<BlockGPU<scalar_t>*>(reinterpret_cast<uint8_t*>(atlas.p_device) + atlas.blocksOffsetBytes);
	
	for(index_t b=0; b<B; ++b){
	DomainGPU<scalar_t> *p_domainCPU = p_domainsCPU + b;
	BlockGPU<scalar_t> *p_blocksCPU = p_blocksCPUall + b*numBlocks;
	
	p_domainCPU->numDims = getSpatialDims();
	p_domainCPU->passiveScalarChannels = getPassiveScalarChannels();
	p_domainCPU->advectionScheme = m_advectionScheme;
	p_domainCPU->numBlocks = blocks.size();
	p_domainCPU->numCells = totalSize;
	p_domainCPU->blocks = p_blocksGPUall + b*numBlocks; //already set correct pointer for GPU version
	
	p_domainCPU->viscosity = viscosity.data_ptr<scalar_t>()[0];
	p_domainCPU->scalarViscosity = hasPassiveScalarViscosity() ? getPassiveScalarViscosityDataPtr<scalar_t>() : nullptr; //viscosity.data_ptr<scalar_t>(); viscosity is a CPU tensor
	p_domainCPU->scalarViscosityStatic = isPassiveScalarViscosityStatic();
	
	p_domainCPU->C.value = batchCSRValuePtr<scalar_t>(C, b);
	p_domainCPU->C.index = C->index.data_ptr<index_t>();
	p_domainCPU->C.row = C->row.data_ptr<index_t>();
	p_domainCPU->Adiag = batchFlatPtr<scalar_t>(A, b, B);
	p_domainCPU->P.value = batchCSRValuePtr<scalar_t>(P, b);
	p_domainCPU->P.index = P->index.data_ptr<index_t>();
	p_domainCPU->P.row = P->row.data_ptr<index_t>();
#ifdef WITH_GRAD
	// the viscosities are shared by all environments, their gradients sum over them
	p_domainCPU->viscosity_grad = getViscosityGradDataPtr<scalar_t>();
	p_domainCPU->scalarViscosity_grad = hasPassiveScalarViscosity() ? getPassiveScalarViscosityGradDataPtr<scalar_t>() : p_domainCPU->viscosity_grad;

	p_domainCPU->C_grad.value = batchCSRValuePtr<scalar_t>(C_grad, b);
	p_domainCPU->C_grad.index = C_grad->index.data_ptr<index_t>();
	p_domainCPU->C_grad.row = C_grad->row.data_ptr<index_t>();
	p_domainCPU->Adiag_grad = batchFlatGradPtr<scalar_t>(A_grad, b, B);
	p_domainCPU->P_grad.value = batchCSRValuePtr<scalar_t>(P_grad, b);
	p_domainCPU->P_grad.index = P_grad->index.data_ptr<index_t>();
	p_domainCPU->P_grad.row = P_grad->row.data_ptr<index_t>();
#endif
	
	if(hasPassiveScalar()){
		p_domainCPU->scalarRHS = batchFlatPtr<scalar_t>(scalarRHS, b, B);
		p_domainCPU->scalarResult = batchFlatPtr<scalar_t>(scalarResult, b, B);
	} else {
		p_domainCPU->scalarRHS = nullptr;
		p_domainCPU->scalarResult = nullptr;
	}
	p_domainCPU->velocityRHS = batchFlatPtr<scalar_t>(velocityRHS, b, B);
	p_domainCPU->velocityResult = batchFlatPtr<scalar_t>(velocityResult, b, B);
	p_domainCPU->pressureRHS = batchFlatPtr<scalar_t>(pressureRHS, b, B);
	p_domainCPU->pressureRHSdiv = batchFlatPtr<scalar_t>(pressureRHSdiv, b, B);
	p_domainCPU->pressureResult = batchFlatPtr<scalar_t>(pressureResult, b, B);

	// Electric potential fields (optional). The potential matrix does not depend on the
	// state and is shared by all environments.
	if(Epot.has_value()){
		p_domainCPU->Epot.value = batchCSRValuePtr<scalar_t>(Epot.value(), b);
		p_domainCPU->Epot.index = Epot.value()->index.data_ptr<index_t>();
		p_domainCPU->Epot.row   = Epot.value()->row.data_ptr<index_t>();
	} else {
		p_domainCPU->Epot.value = nullptr;
		p_domainCPU->Epot.index = nullptr;
		p_domainCPU->Epot.row   = nullptr;
	}
	p_domainCPU->epotRHS    = batchFlatPtr<scalar_t>(epotRHS, b, B);
	p_domainCPU->epotResult = batchFlatPtr<scalar_t>(epotResult, b, B);


#ifdef WITH_GRAD
	if(hasPassiveScalar()){
		p_domainCPU->scalarRHS_grad = batchFlatGradPtr<scalar_t>(scalarRHS_grad, b, B);
		p_domainCPU->scalarResult_grad = batchFlatGradPtr<scalar_t>(scalarResult_grad, b, B);
	} else {
		p_domainCPU->scalarRHS_grad = nullptr;
		p_domainCPU->scalarResult_grad = nullptr;
	}
	p_domainCPU->velocityRHS_grad = batchFlatGradPtr<scalar_t>(velocityRHS_grad, b, B);
	p_domainCPU->velocityResult_grad = batchFlatGradPtr<scalar_t>(velocityResult_grad, b, B);
	p_domainCPU->pressureRHS_grad = batchFlatGradPtr<scalar_t>(pressureRHS_grad, b, B);
	p_domainCPU->pressureRHSdiv_grad = batchFlatGradPtr<scalar_t>(pressureRHSdiv_grad, b, B);
	p_domainCPU->pressureResult_grad = batchFlatGradPtr<scalar_t>(pressureResult_grad, b, B);
	p_domainCPU->epotResult_grad = hasEpotResultGrad()
		? batchFlatPtr<scalar_t>(epotResult_grad, b, B) : nullptr;
#endif
	
	for(index_t blockIdx=0; blockIdx<numBlocks; ++blockIdx){
		BlockGPU<scalar_t> *p_blockCPU = p_blocksCPU + blockIdx;
		std::shared_ptr<const Block> block = blocks[blockIdx];
		
		TORCH_CHECK(block->hasPassiveScalar()==hasPassiveScalar(), "Inconsistent use of passive scalar.")
		TORCH_CHECK(block->getPassiveScalarChannels()==getPassiveScalarChannels(), "Inconsistent number of channels in passive scalar.")
		TORCH_CHECK(block->getBatchSize()==B, "All blocks must have the same batch size.")
		
		p_blockCPU->globalOffset = block->globalOffset; //offset in cells from first block start. used e.g. for CSR indices
		p_blockCPU->csrOffset = block->csrOffset;
		p_blockCPU->size = block->getSizes();
		p_blockCPU->stride = block->getStrides();
		p_blockCPU->viscosity = batchSlicePtr<scalar_t>(block->m_viscosity, b, B, "block viscosity");
		p_blockCPU->isViscosityStatic = block->isViscosityStatic();
		p_blockCPU->velocity = batchSlicePtr<scalar_t>(block->velocity, b, B, "velocity");
		p_blockCPU->velocitySource = batchSlicePtr<scalar_t>(block->velocitySource, b, B, "velocity source");
		p_blockCPU->isVelocitySourceStatic = block->velocitySourceStatic;
		p_blockCPU->pressure = batchSlicePtr<scalar_t>(block->pressure, b, B, "pressure");
		p_blockCPU->scalarData = batchSlicePtr<scalar_t>(block->passiveScalar, b, B, "passive scalar");
#ifdef WITH_GRAD
		p_blockCPU->viscosity_grad = batchGradSlicePtr<scalar_t>(block->m_viscosity_grad, b, B, "block viscosity grad");
		p_blockCPU->velocity_grad = batchGradSlicePtr<scalar_t>(block->velocity_grad, b, B, "velocity grad");
		p_blockCPU->velocitySource_grad = batchGradSlicePtr<scalar_t>(block->velocitySource_grad, b, B, "velocity source grad");
		p_blockCPU->pressure_grad = batchGradSlicePtr<scalar_t>(block->pressure_grad, b, B, "pressure grad");
		p_blockCPU->scalarData_grad = batchGradSlicePtr<scalar_t>(block->passiveScalar_grad, b, B, "passive scalar grad");
#endif
		for(index_t boundIdx=0; boundIdx<getSpatialDims()*2; ++boundIdx){
			const std::shared_ptr<Boundary> boundary = block->boundaries[boundIdx];
			p_blockCPU->boundaries[boundIdx].type = boundary->type;
			switch(boundary->type){
			case BoundaryType::FIXED:
			{
				std::shared_ptr<const FixedBoundary> bound = std::static_pointer_cast<const FixedBoundary>(boundary);
				
				TORCH_CHECK(block->hasPassiveScalar()==bound->hasPassiveScalar(), "Inconsistent use of passive scalar.")
				TORCH_CHECK(block->getPassiveScalarChannels()==bound->getPassiveScalarChannels(), "Inconsistent number of channels in passive scalar.")
				
				FixedBoundaryGPU<scalar_t> fb;
				memset(&fb, 0, sizeof(FixedBoundaryGPU<scalar_t>));
				fb.size = bound->getSizes();
				fb.stride = bound->getStrides();
				
				if(bound->isPassiveScalarBoundaryTypeStatic()){
					fb.passiveScalar.boundaryType = bound->m_passiveScalarTypes ? bound->m_passiveScalarTypes.value()[0] : BoundaryConditionType::DIRICHLET;
					fb.passiveScalar.isStaticType = true;
				}else{
					fb.passiveScalar.p_boundaryTypes = reinterpret_cast<BoundaryConditionType*>(
						batchSlicePtr<BoundaryConditionType_base_type>(bound->m_passiveScalarTypes_tensor, b, B, "boundary passive scalar types"));
					fb.passiveScalar.isStaticType = false;
				}
				// boundary data of batch size 1 is shared by all environments
				fb.passiveScalar.data = batchSlicePtr<scalar_t>(bound->m_passiveScalar, b, B, "boundary passive scalar");
				fb.passiveScalar.isStatic = bound->m_passiveScalarStatic;
				
				fb.velocity.boundaryType = bound->m_velocityType;
				fb.velocity.isStaticType = true;
				fb.velocity.data = batchSlicePtr<scalar_t>(bound->m_velocity, b, B, "boundary velocity");
				fb.velocity.isStatic = bound->m_velocityStatic;
				
				fb.pressure.boundaryType = bound->m_pressureType;
				fb.pressure.isStaticType = true;
				fb.pressure.data = batchSlicePtr<scalar_t>(bound->m_pressure, b, B, "boundary pressure");
				fb.pressure.isStatic = bound->m_pressureStatic;
				
#ifdef WITH_GRAD
				fb.passiveScalar.grad = batchGradSlicePtr<scalar_t>(bound->m_passiveScalar_grad, b, B, "boundary passive scalar grad");
				fb.velocity.grad = batchGradSlicePtr<scalar_t>(bound->m_velocity_grad, b, B, "boundary velocity grad");
				fb.pressure.grad = nullptr;
#endif
				
				fb.hasTransform = bound->hasTransform();
				fb.transform = bound->getTransformDataPtr<scalar_t>();

				// Electric potential BC (MHD)
				fb.potentialType = bound->getPotentialBC();
				fb.potentialCw = static_cast<scalar_t>(bound->getPotentialCw());
				{
					I4 faceSize = block->getSizes();
					faceSize.a[boundIdx>>1] = 1;
					fb.potentialStride = {{.x=1, .y=faceSize.x, .z=faceSize.x*faceSize.y, .w=faceSize.x*faceSize.y*faceSize.z}};
					const index_t spatialDims = getSpatialDims();
					auto checkFaceTensor = [&](const torch::Tensor &t, const char *what){
						bool shapeMatches = t.dim()==spatialDims+2;
						for(index_t dim=0; shapeMatches && dim<spatialDims; ++dim){
							shapeMatches = t.size(t.dim()-1-dim)==faceSize.a[dim]; // NCDHW: x is last
						}
						TORCH_CHECK(shapeMatches, "Potential ", what, " of block '", block->name, "' face ", boundIdx,
							" do not match the face's cell grid.");
						TORCH_CHECK(t.device()==getDevice(), "Potential ", what, " must be on the domain's device.");
					};
					fb.potentialTypes = nullptr;
					if(bound->hasPotentialTypes()){
						const torch::Tensor &types = bound->m_potentialTypes.value();
						checkFaceTensor(types, "BC types");
						// the types set the Epot matrix, which all environments share
						TORCH_CHECK(types.size(0)==1, "Potential BC types must be shared by all environments (batch size 1).");
						fb.potentialTypes = types.data_ptr<PotentialBC_base_type>();
					}
					fb.potentialValues = nullptr;
					if(bound->hasPotentialValues()){
						const torch::Tensor &values = bound->m_potentialValues.value();
						checkFaceTensor(values, "values");
						TORCH_CHECK(values.scalar_type()==getDtype(), "Potential values must have the domain's dtype.");
						fb.potentialValues = batchSlicePtr<scalar_t>(values, b, B, "potential values");
					}
#ifdef WITH_GRAD
					fb.potentialValuesGrad = nullptr;
					if(bound->hasPotentialValuesGrad()){
						const torch::Tensor &grad = bound->m_potentialValuesGrad.value();
						checkFaceTensor(grad, "values grad");
						// a shared slice would be written by every environment at once
						TORCH_CHECK(grad.size(0)==B, "Potential values grad must hold one slice per environment.");
						fb.potentialValuesGrad = batchSlicePtr<scalar_t>(grad, b, B, "potential values grad");
					}
#endif //WITH_GRAD
				}

				p_blockCPU->boundaries[boundIdx].fb = fb;
				break;
			}
			case BoundaryType::DIRICHLET:
			{
				std::shared_ptr<const StaticDirichletBoundary> bound = std::static_pointer_cast<const StaticDirichletBoundary>(boundary);
				StaticDirichletBoundaryGPU<scalar_t> sdb;
				memset(&sdb, 0, sizeof(StaticDirichletBoundaryGPU<scalar_t>));
				sdb.slip = bound->slip.data_ptr<scalar_t>()[0];
				memcpy(&sdb.velocity.a, bound->boundaryVelocity.data_ptr<scalar_t>(), sizeof(scalar_t)*getSpatialDims());
				sdb.scalar = bound->boundaryScalar.data_ptr<scalar_t>()[0];
#ifdef WITH_GRAD
				sdb.velocity_grad = getTensorDataPtr<scalar_t>(bound->boundaryVelocity_grad);
				sdb.scalar_grad = getTensorDataPtr<scalar_t>(bound->boundaryScalar_grad);
#endif
				p_blockCPU->boundaries[boundIdx].sdb = sdb;
				break;
			}
			case BoundaryType::DIRICHLET_VARYING:
			{
				std::shared_ptr<const VaryingDirichletBoundary> bound = std::static_pointer_cast<const VaryingDirichletBoundary>(boundary);
				VaryingDirichletBoundaryGPU<scalar_t> vdb;
				memset(&vdb, 0, sizeof(VaryingDirichletBoundaryGPU<scalar_t>));
				vdb.slip = bound->slip.data_ptr<scalar_t>()[0];
				vdb.velocity = bound->boundaryVelocity.data_ptr<scalar_t>();
				vdb.scalar = bound->boundaryScalar.data_ptr<scalar_t>();
				vdb.size = bound->getSizes();
				vdb.stride = bound->getStrides();
#ifdef WITH_GRAD
				vdb.velocity_grad = getTensorDataPtr<scalar_t>(bound->boundaryVelocity_grad);
				vdb.scalar_grad = getTensorDataPtr<scalar_t>(bound->boundaryScalar_grad);
#endif
				vdb.hasTransform = bound->hasTransform;
				vdb.transform = bound->hasTransform ? bound->transform.data_ptr<scalar_t>() : nullptr;
				p_blockCPU->boundaries[boundIdx].vdb = vdb;
				break;
			}
			case BoundaryType::NEUMANN:
			{
				StaticNeumannBoundaryGPU<scalar_t> snb;
				memset(&snb, 0, sizeof(StaticNeumannBoundaryGPU<scalar_t>));
				p_blockCPU->boundaries[boundIdx].snb = snb;
				break;
			}
			case BoundaryType::CONNECTED_GRID:
			{
				std::shared_ptr<const ConnectedBoundary> bound = std::static_pointer_cast<const ConnectedBoundary>(boundary);
				ConnectedBoundaryGPU<scalar_t> cb;
				memset(&cb, 0, sizeof(ConnectedBoundaryGPU<scalar_t>));
				cb.connectedGridIndex = getBlockIdx(bound->getConnectedBlock());
				const index_t numAxes = static_cast<index_t>(bound->axes.size());
				for(index_t axisIdx=0; axisIdx<numAxes; ++axisIdx){
					cb.axes.a[axisIdx] = bound->axes[axisIdx];
				}
				p_blockCPU->boundaries[boundIdx].cb = cb;
				break;
			}
			case BoundaryType::PERIODIC:
			{
				PeriodicBoundaryGPU<scalar_t> pb;
				memset(&pb, 0, sizeof(PeriodicBoundaryGPU<scalar_t>));
				p_blockCPU->boundaries[boundIdx].pb = pb;
				break;
			}
			default:
				TORCH_CHECK(false, "Unknown boundary encountered in Domain::UpdateDomainGPU.");
				break;
			}
		}
		// the grid is shared by all environments
		p_blockCPU->hasTransform = block->hasTransform();
		p_blockCPU->transform = block->getTransformDataPtr<scalar_t>();
		p_blockCPU->hasFaceTransform = block->hasFaceTransform();
		p_blockCPU->faceTransform = block->getFaceTransformDataPtr<scalar_t>();
	}
	} // environments
	
	CopyDomainToGPU(atlas);
	//set the pointers for host memory
	for(index_t b=0; b<B; ++b){
		p_domainsCPU[b].blocks = p_blocksCPUall + b*numBlocks;
	}
	
	setTensorChanged(false);
}

index_t Domain::getBatchSize() const {
	return blocks.empty() ? 1 : blocks[0]->getBatchSize();
}

void Domain::setBatchSize(const index_t batchSize){
	TORCH_CHECK(batchSize>=1, "Batch size must be at least 1.");
	const index_t current = getBatchSize();
	if(batchSize==current) return;
	TORCH_CHECK(current==1, "The batch size can only be changed from 1 (current batch size: " + std::to_string(current) + ").");
	auto repeatBatch = [batchSize](const torch::Tensor &t){
		std::vector<int64_t> reps(t.dim(), 1);
		reps[0] = batchSize;
		return t.repeat(reps).contiguous();
	};
	for(auto block : blocks){
		block->velocity = repeatBatch(block->velocity);
		block->pressure = repeatBatch(block->pressure);
		if(block->passiveScalar) block->passiveScalar = repeatBatch(block->passiveScalar.value());
		if(block->epot) block->epot = repeatBatch(block->epot.value());
		block->isTensorChanged = true;
	}
	isTensorChanged = true;
	// the solve vectors, matrices and the device atlas depend on the batch size
	initialized = false;
}

index_t Domain::getBlockIdx(const std::shared_ptr<const Block> block) const {
	auto it = find(blocks.begin(), blocks.end(), block);
	if(it==blocks.end()) return -1;
	return it - blocks.begin();
}

torch::Dtype Domain::getDtype() const {
	return m_dtype;
    //TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
    //return blocks[0]->getDtype();
}

py::object Domain::getPyDtype() const {
	//return (PyObject*)torch::getTHPDtype(getDtype());
	if(pyDtype.has_value()){
		return pyDtype.value();
	} else {
		TORCH_CHECK(false, "No python dtype set.");
	}
}

torch::Device Domain::getDevice() const {
	return m_device;
    //TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
    //return blocks[0]->getDevice();
}

torch::TensorOptions Domain::getTensorOptions() const {
	return valueOptions;
}

index_t Domain::getSpatialDims() const {
	return m_spatialDims;
    //TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
    //return blocks[0]->getSpatialDims();
}
index_t Domain::getTotalSize() const {
	TORCH_CHECK(initialized, "domain is not initialized.");
	return totalSize;
}

index_t Domain::getMaxBlockSize() const {
    TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
	index_t maxSize = 0;
	for(auto p_block : blocks){
		const index_t blockSize = p_block->getStrides().w;
		if(blockSize>maxSize){
			maxSize = blockSize;
		}
	}
	return maxSize;
}

bool Domain::IsTensorChanged() const {
	if(isTensorChanged){ return true; }
	if(C->IsTensorChanged()){ return true; }
	if(P->IsTensorChanged()){ return true; }
#ifdef WITH_GRAD
	if(C_grad->IsTensorChanged()){ return true; }
	if(P_grad->IsTensorChanged()){ return true; }
#endif
	for(auto block : blocks){
		if(block->IsTensorChanged()){ return true; }
	}
	return false;
}
void Domain::setTensorChanged(const bool changed){
	isTensorChanged = changed;
	C->setTensorChanged(changed);
	P->setTensorChanged(changed);
#ifdef WITH_GRAD
	C_grad->setTensorChanged(changed);
	P_grad->setTensorChanged(changed);
#endif
	for(auto block : blocks){
		block->setTensorChanged(changed);
	}
	
}

std::string Domain::ToString() const {
	std::ostringstream repr;
	repr << "Domain(\"" << name << "\" ";
	//if(blocks.size()>0) repr << getSpatialDims();
	//else repr << "?";
	repr << getSpatialDims() << "D";
	repr << ", scalarChannels=" << getPassiveScalarChannels();
	repr << ", blocks=[";
	//for(auto block : blocks){
	for(auto blockIt = blocks.begin(); blockIt!=blocks.end(); ++blockIt){
		repr << (*blockIt)->name;
		if(blockIt!=(blocks.end()-1)) repr << ", ";
	}
	repr << "], initialized=" << initialized << " )";
	return repr.str();
	
}

