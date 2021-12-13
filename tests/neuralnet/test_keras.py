import tensorflow.keras as keras
import pytest
import pyomo.environ as pyo
import tensorflow

from omlt.block import OmltBlock
from omlt.io.keras_reader import load_keras_sequential
from omlt.neuralnet import NeuralNetworkFormulation
from omlt.scaling import OffsetScaling

from conftest import get_neural_network_data

#TODO: work on names
FullSpaceContinuousFormulation = NeuralNetworkFormulation

def _test_keras_linear_131(keras_fname, reduced_space=False):
    x, y, x_test = get_neural_network_data("131")

    nn = tensorflow.keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn, input_bounds=[(-1,1)])
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    if reduced_space:
        formulation = ReducedSpaceContinuousFormulation(net)
    else:
        formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("ipopt").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def _test_keras_mip_relu_131(keras_fname):
    x, y, x_test = get_neural_network_data("131")

    nn = tensorflow.keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn,input_bounds = [(-1,1)])

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReLUBigMFormulation(net)
    m.neural_net_block.build_formulation(formulation)
    m.obj = pyo.Objective(expr=0)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("cbc").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def _test_keras_complementarity_relu_131(keras_fname):
    x, y, x_test = get_neural_network_data("131")

    nn = tensorflow.keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReLUComplementarityFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("ipopt").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def _test_keras_linear_big(keras_fname, reduced_space=False):
    x, y, x_test = get_neural_network_data("131")

    nn = keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    if reduced_space:
        formulation = ReducedSpaceContinuousFormulation(net)
    else:
        formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("ipopt").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def test_keras_linear_131_full(datadir):
    _test_keras_linear_131(datadir.file("keras_linear_131"))
    _test_keras_linear_131(datadir.file("keras_linear_131_sigmoid"))
    _test_keras_linear_131(datadir.file("keras_linear_131_sigmoid_output_activation"))
    _test_keras_linear_131(
        datadir.file("keras_linear_131_sigmoid_softplus_output_activation")
    )


@pytest.mark.skip("reduced space not completed")
def test_keras_linear_131_reduced(datadir):
    _test_keras_linear_131(datadir.file("keras_linear_131"), reduced_space=True)
    _test_keras_linear_131(
        datadir.file("keras_linear_131_sigmoid"),
        reduced_space=True,
    )
    _test_keras_linear_131(
        datadir.file("keras_linear_131_sigmoid_output_activation"),
        reduced_space=True,
    )
    _test_keras_linear_131(
        datadir.file("keras_linear_131_sigmoid_softplus_output_activation"),
        reduced_space=True,
    )

@pytest.mark.skip("relu is done a different way now")
def test_keras_linear_131_relu(datadir):
    _test_keras_mip_relu_131(
        datadir.file("keras_linear_131_relu"),
    )
    _test_keras_complementarity_relu_131(
        datadir.file("keras_linear_131_relu"),
    )

def test_keras_linear_big(datadir):
    _test_keras_linear_big(datadir.file("big"), reduced_space=False)
    # too slow
    # _test_keras_linear_big('./models/big', reduced_space=True)

#if __name__ ==' __main__':
#    test_keras_linear_131_full()
