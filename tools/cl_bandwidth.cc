#include "ppl/common/log.h"
#include "ppl/common/destructor.h"
#include "ppl/common/ocl/device.h"
#include "ppl/common/ocl/framechain.h"
#include "ppl/common/ocl/runkernel.h"
#include "ppl/common/ocl/kernelpool.h"

#include "CL/opencl.h"

#include "kernels/flush.cl.h"
#include "kernels/demo.cl.h"

#include <string>

#define BUFFER_SIZE 500*1024*1024

static ppl::common::RetCode demo(
    ppl::common::ocl::FrameChain* frame_chain)
{
    cl_int ret = 0;
    std::string options;
    frame_chain->setCompileOptions(options.c_str());
    SET_PROGRAM_SOURCE(frame_chain, demo);
    size_t gs[] = {100};
    int a = 1;
    runOclKernel(frame_chain, "demo", 1, gs, nullptr, a);
    cl_int rett = clFinish(frame_chain->getQueue());
    if (rett) { 
        LOG(ERROR) << "clFinsh failed.";
    }

    uint64_t time_ns = frame_chain->getKernelTime();
    LOG(ERROR) << time_ns;

    return ppl::common::RC_SUCCESS;
}

int main() {
    ppl::common::ocl::createSharedFrameChain(true);
    ppl::common::ocl::FrameChain * frame_chain = ppl::common::ocl::getSharedFrameChain();
    frame_chain->setTuningQueueStatus(false);
    frame_chain->setProjectName("cl_bandwidth");
    ppl::common::RetCode rc = demo(frame_chain);

    ppl::common::ocl::removeAllKernelsFromPool();
    
    return 0;
}
