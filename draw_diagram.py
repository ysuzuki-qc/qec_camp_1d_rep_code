import os
import stim
from lib_1drep.data import Circuit
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.qec import _convert_circuit_to_stim
from lib_1drep.noise import create_1drep_uniform_noise_model

if not os.path.exists("./diagram"):
    os.mkdir("./diagram/")


def draw_diagram(circuit: Circuit):
    noise_model = create_1drep_uniform_noise_model(1e-2, circuit.code_distance)
    stim_dem_circuit: stim.Circuit = _convert_circuit_to_stim(circuit, noise_model)
    # stim_dem_circuit = stim_circuit.detector_error_model()
    with open("./diagram/stim_timeline.svg", "w") as f:
        diagram = stim_dem_circuit.diagram("timeline-svg")
        f.write(str(diagram))
    with open("./diagram/stim_match_graph.svg", "w") as f:
        diagram = stim_dem_circuit.diagram("match-graph-svg")
        f.write(str(diagram))
    with open("./diagram/stim_detector_slice.svg", "w") as f:
        diagram = stim_dem_circuit.diagram("detector-slice-svg")
        f.write(str(diagram))
    with open("./diagram/stim_detector_slice_with_ops.svg", "w") as f:
        diagram = stim_dem_circuit.diagram("detslice-with-ops-svg")
        f.write(str(diagram))
    with open("./diagram/dem.txt", "w") as fout:
        stim_dem_circuit.detector_error_model().to_file(fout)


code_distance = 5
num_round = 5
initial_state = 0
circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
draw_diagram(circuit)
