import pyomo.environ as pe
import onnx
import onnxruntime as ort
import numpy as np
from omlt import OffsetScaling
from omlt.io.onnx import load_onnx_neural_network
from omlt.block import OmltBlock
from omlt.neuralnet import ReLUBigMFormulation, FullSpaceContinuousFormulation


# mod_name = '../calvin-nn/calvin_test_please_ignore.onnx'
mod_name = '../calvin-nn/calvin_cnn.onnx'
# mod_name = '../calvin-nn/calvin_dense.onnx'
# mod_name = 'tests/models/gemm.onnx'
# mod_name = 'tests/models/keras_linear_131_relu.onnx'
# mod_name = 'tests/models/keras_linear_131_sigmoid.onnx'
# mod_name = '../calvin-nn/2d_input.onnx'
# mod_name = 'calvin_dense.onnx'
mod = onnx.load(mod_name)


# input_bounds = [(0, 1e-3) for _ in range(28*28)]
input_bounds = [(0, 1e-5) for _ in range(28*28)]
# input_bounds = [(0, 1e-3) for _ in range(7*7)]
# input_bounds = [(-100, 100) for _ in range(100)]
scaling = OffsetScaling(
    offset_inputs=[0 for _ in input_bounds],
    factor_inputs=[1 for _ in input_bounds],
    offset_outputs=[],
    factor_outputs=[],
)
# net = load_onnx_neural_network(mod, input_bounds=input_bounds, scaling_object=scaling)
net = load_onnx_neural_network(mod, input_bounds=input_bounds)
# print(list(net.nodes))

if False:
    orig_x = np.random.uniform(size=(1, 7, 7))
    # orig_x = np.random.uniform(size=784)
    # orig_x = np.random.uniform(size=(3, 5))
    # orig_x = np.random.uniform(size=(1))
    x = orig_x.copy()
    nodes = list(net.nodes)
    for node in nodes:
        x = node.eval(x)
    print('eval ', x)

    ort_session = ort.InferenceSession(mod_name)
    print('ort ', ort_session.run(None, {'input': [orig_x.astype(np.float32)]}))


if True:
    m = pe.ConcreteModel()
    m.nn = OmltBlock()
    # formulation = ReLUBigMFormulation(net)
    formulation = ReLUBigMFormulation(net)
    m.nn.build_formulation(formulation)
    m.nn.outputs.pprint()
    # m.o = pe.Objective(expr=m.nn.layer[0].z[0])
    # m.o = pe.Objective(expr=m.nn.outputs[0])
    m.o = pe.Objective(expr=m.nn.outputs[0, 0])
    # m.o = pe.Objective(expr=0)

    if False:
        # m.nn.layer[1].deactivate()
        del m.nn.outputs
        del m.nn.output_assignment
        del m.nn.layer[2]
        del m.nn.layer[3]

    if True:
        # s = pe.SolverFactory('gurobi_direct')
        s = pe.SolverFactory('gurobi')
        res = s.solve(m, tee=True)
        print(res)

    if False:
        m.nn.pprint()

    # use one of the output nodes as objective and maximize it