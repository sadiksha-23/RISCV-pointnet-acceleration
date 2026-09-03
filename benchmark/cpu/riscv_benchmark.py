import sys

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.cachehierarchies.classic.private_l1_shared_l2_cache_hierarchy import (
    PrivateL1SharedL2CacheHierarchy,
)
from gem5.resources.resource import CustomResource
from gem5.simulate.simulator import Simulator
from gem5.isas import ISA


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <RISC-V binary>")
    sys.exit(1)

binary_path = sys.argv[1]


# Single-core out-of-order RISC-V processor
processor = SimpleProcessor(
    cpu_type=CPUTypes.O3,
    isa=ISA.RISCV,
    num_cores=1,
)

# Modeled out-of-order RISC-V core with RVV support
core = processor.get_cores()[0].core

core.fetchWidth = 3
core.decodeWidth = 3
core.renameWidth = 3
core.issueWidth = 6
core.dispatchWidth = 6
core.wbWidth = 6
core.commitWidth = 6

core.numPhysIntRegs = 128
core.numPhysFloatRegs = 128
core.numPhysVecRegs = 256


# Private 32 KiB L1 instruction and data caches with a shared 512 KiB L2
cache_hierarchy = PrivateL1SharedL2CacheHierarchy(
    l1i_size="32kB",
    l1i_assoc=4,
    l1d_size="32kB",
    l1d_assoc=8,
    l2_size="512kB",
    l2_assoc=8,
)


# Single-channel 8 GiB DDR4-2400 memory
memory = SingleChannelDDR4_2400(
    size="8GB",
)


# System board running at 2 GHz
board = SimpleBoard(
    clk_freq="2GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)


# Run the binary in syscall-emulation mode
board.set_se_binary_workload(
    CustomResource(binary_path),
)


simulator = Simulator(board=board)

print(f"Starting RISC-V simulation: {binary_path}")
simulator.run()
print("RISC-V simulation finished successfully.")