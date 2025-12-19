import FourierNeuralOperators as FNO
import OptimalTransportEncoding as OTE

using FourierNeuralOperators: OptimalTransportNeuralOperator, evaluate_dataset_mape
using OptimalTransportEncoding: OTEDataSample
using Lux
using Serialization

include("utils.jl")

function load(path::AbstractString)
    object = open(path, "r") do io
        deserialize(io)
    end
    return object
end

otno_model_path = normpath(joinpath(@__DIR__, "pretrained_otno_weights/otno_weights_epoch_147.jls"))
dataset_dir = normpath(joinpath(@__DIR__, "..", "datasets/ShapeNet-Car"))

(params, st) = load(otno_model_path)
states = Lux.testmode(st)

(dataset_train, dataset_val, dataset_test) = load_datasets(dataset_dir, ".jls")
(xs_test, ys_test) = dataset_test

# TODO: save the model instance during the checkpointing saves as well
D = 2
channels_in = 9
channels_hidden = channels_in
channels_out = 1
modes = (16, 16, 16, 16)
rank_ratio = 0.5f0

model = FNO.OptimalTransportNeuralOperator{D}(
    channels_in, channels_hidden, channels_out; modes, rank_ratio
)
display(model)

mrl2e = evaluate_dataset_mrl2e(model, params, states, (xs_test, ys_test))
mrl2e = evaluate_dataset_mrl2e(model, params, states, (dataset_train.xs, dataset_train.ys))
mrl2e = evaluate_dataset_mrl2e(model, params, states, (dataset_val.xs, dataset_val.ys))
