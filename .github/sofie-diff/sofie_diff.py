"""Differential test: SOFIE-generated inference vs onnxruntime on small single-operator ONNX models."""
import os, sys, traceback
import numpy as np
import onnx
from onnx import TensorProto, helper as h
import onnxruntime as ort
import ROOT

SOFIE = ROOT.TMVA.Experimental.SOFIE
rng = np.random.default_rng(0)
F = TensorProto.FLOAT
I64 = TensorProto.INT64


def f32(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def model(name, nodes, inputs, outputs, inits=(), opset=17):
    g = h.make_graph(nodes, name, inputs, outputs, initializer=list(inits))
    m = h.make_model(g, opset_imports=[h.make_opsetid("", opset)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def vi(n, t, shape):
    return h.make_tensor_value_info(n, t, shape)


def init_i64(n, vals):
    return h.make_tensor(n, I64, [len(vals)], vals)


CASES = []


def case(name, nodes, inputs, outputs, feeds, inits=(), opset=17):
    CASES.append((name, nodes, inputs, outputs, feeds, inits, opset))


X = f32(2, 3, 4)
x_in = [vi("X", F, [2, 3, 4])]
y_out = lambda shape: [vi("Y", F, shape)]

case("Softmax_opset11_default_axis", [h.make_node("Softmax", ["X"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X}, opset=11)
case("Softmax_opset13_default_axis", [h.make_node("Softmax", ["X"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X}, opset=13)
case("LogSoftmax_axis1", [h.make_node("LogSoftmax", ["X"], ["Y"], axis=1)], x_in, y_out([2, 3, 4]), {"X": X})
case("LeakyRelu_default", [h.make_node("LeakyRelu", ["X"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X})
case("Elu_alpha", [h.make_node("Elu", ["X"], ["Y"], alpha=0.5)], x_in, y_out([2, 3, 4]), {"X": X})
case("HardSigmoid", [h.make_node("HardSigmoid", ["X"], ["Y"], alpha=0.3, beta=0.4)], x_in, y_out([2, 3, 4]), {"X": X})
case("HardSwish", [h.make_node("HardSwish", ["X"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X * 4}, opset=14)
case("Selu_default", [h.make_node("Selu", ["X"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X})
case("Gelu_tanh", [h.make_node("Gelu", ["X"], ["Y"], approximate="tanh")], x_in, y_out([2, 3, 4]), {"X": X}, opset=20)
case("ThresholdedRelu", [h.make_node("ThresholdedRelu", ["X"], ["Y"], alpha=0.2)], x_in, y_out([2, 3, 4]), {"X": X})
case("Transpose_default_perm", [h.make_node("Transpose", ["X"], ["Y"])], x_in, y_out([4, 3, 2]), {"X": X})
case("Transpose_perm", [h.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])], x_in, y_out([3, 2, 4]), {"X": X})
case("Concat_neg_axis", [h.make_node("Concat", ["X", "X"], ["Y"], axis=-1)], x_in, y_out([2, 3, 8]), {"X": X})
case("Flatten_axis2", [h.make_node("Flatten", ["X"], ["Y"], axis=2)], x_in, y_out([6, 4]), {"X": X})
case("Flatten_axis0", [h.make_node("Flatten", ["X"], ["Y"], axis=0)], x_in, y_out([1, 24]), {"X": X})
case("Clip_min_only", [h.make_node("Clip", ["X", "mn"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X},
     inits=[h.make_tensor("mn", F, [], [0.1])], opset=13)
case("Clip_max_only", [h.make_node("Clip", ["X", "", "mx"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X},
     inits=[h.make_tensor("mx", F, [], [0.2])], opset=13)
case("ReduceMean_axes_keep0", [h.make_node("ReduceMean", ["X", "ax"], ["Y"], keepdims=0)], x_in, y_out([2, 4]), {"X": X},
     inits=[init_i64("ax", [1])], opset=18)
case("ReduceMean_noaxes", [h.make_node("ReduceMean", ["X"], ["Y"], keepdims=1)], x_in, y_out([1, 1, 1]), {"X": X}, opset=18)
case("ReduceSum_neg_axes", [h.make_node("ReduceSum", ["X", "ax"], ["Y"], keepdims=1)], x_in, y_out([2, 3, 1]), {"X": X},
     inits=[init_i64("ax", [-1])], opset=13)
case("ReduceMax_multi_axes", [h.make_node("ReduceMax", ["X", "ax"], ["Y"], keepdims=0)], x_in, y_out([3]), {"X": X},
     inits=[init_i64("ax", [0, 2])], opset=18)
case("ReduceProd_axis0", [h.make_node("ReduceProd", ["X", "ax"], ["Y"], keepdims=1)], x_in, y_out([1, 3, 4]), {"X": X},
     inits=[init_i64("ax", [0])], opset=18)
case("ReduceSumSquare", [h.make_node("ReduceSumSquare", ["X", "ax"], ["Y"], keepdims=0)], x_in, y_out([2, 3]), {"X": X},
     inits=[init_i64("ax", [2])], opset=18)
case("ArgMax_keep0", [h.make_node("ArgMax", ["X"], ["Y"], axis=1, keepdims=0)], x_in, [vi("Y", I64, [2, 4])], {"X": X})
case("ArgMax_last_index", [h.make_node("ArgMax", ["X"], ["Y"], axis=2, keepdims=1, select_last_index=1)], x_in,
     [vi("Y", I64, [2, 3, 1])], {"X": np.round(X).astype(np.float32)})
case("Gather_neg_indices", [h.make_node("Gather", ["X", "idx"], ["Y"], axis=1)], x_in, y_out([2, 2, 4]), {"X": X},
     inits=[init_i64("idx", [-1, 0])])
case("Slice_neg_step", [h.make_node("Slice", ["X", "st", "en", "ax", "sp"], ["Y"])], x_in, y_out([2, 3, 2]), {"X": X},
     inits=[init_i64("st", [-1]), init_i64("en", [-5]), init_i64("ax", [2]), init_i64("sp", [-2])])
case("Split_uneven", [h.make_node("Split", ["X", "sp"], ["A", "B"], axis=2)], x_in,
     [vi("A", F, [2, 3, 1]), vi("B", F, [2, 3, 3])], {"X": X}, inits=[init_i64("sp", [1, 3])], opset=13)
case("Pad_reflect", [h.make_node("Pad", ["X", "pads"], ["Y"], mode="reflect")], x_in, y_out([2, 3, 6]), {"X": X},
     inits=[init_i64("pads", [0, 0, 1, 0, 0, 1])], opset=13)
case("Pad_edge", [h.make_node("Pad", ["X", "pads"], ["Y"], mode="edge")], x_in, y_out([2, 5, 4]), {"X": X},
     inits=[init_i64("pads", [0, 1, 0, 0, 1, 0])], opset=13)
case("TopK_smallest", [h.make_node("TopK", ["X", "k"], ["V", "I"], axis=-1, largest=0)], x_in,
     [vi("V", F, [2, 3, 2]), vi("I", I64, [2, 3, 2])], {"X": X}, inits=[init_i64("k", [2])])
case("Tile", [h.make_node("Tile", ["X", "r"], ["Y"])], x_in, y_out([2, 6, 4]), {"X": X}, inits=[init_i64("r", [1, 2, 1])])
case("Expand", [h.make_node("Expand", ["X", "s"], ["Y"])], [vi("X", F, [3, 1])], y_out([2, 3, 4]), {"X": f32(3, 1)},
     inits=[init_i64("s", [2, 1, 4])])
case("Where_broadcast", [h.make_node("Greater", ["X", "Z"], ["C"]), h.make_node("Where", ["C", "X", "Z"], ["Y"])],
     [vi("X", F, [2, 3, 4]), vi("Z", F, [4])], y_out([2, 3, 4]), {"X": X, "Z": f32(4)})
case("Mean_nary", [h.make_node("Mean", ["X", "Z", "W"], ["Y"])], [vi("X", F, [2, 3, 4]), vi("Z", F, [4]), vi("W", F, [3, 1])],
     y_out([2, 3, 4]), {"X": X, "Z": f32(4), "W": f32(3, 1)})
case("Pow_broadcast", [h.make_node("Pow", ["X", "Z"], ["Y"])], [vi("X", F, [2, 3, 4]), vi("Z", F, [4])], y_out([2, 3, 4]),
     {"X": np.abs(X) + 0.1, "Z": f32(4)})
case("LayerNorm_axis1", [h.make_node("LayerNormalization", ["X", "s", "b"], ["Y"], axis=1, epsilon=1e-3)], x_in,
     y_out([2, 3, 4]), {"X": X}, inits=[h.make_tensor("s", F, [3, 4], f32(3, 4).ravel()), h.make_tensor("b", F, [3, 4], f32(3, 4).ravel())])
case("InstanceNorm", [h.make_node("InstanceNormalization", ["X", "s", "b"], ["Y"], epsilon=1e-3)], x_in, y_out([2, 3, 4]),
     {"X": X}, inits=[h.make_tensor("s", F, [3], [1.0, 2.0, 0.5]), h.make_tensor("b", F, [3], [0.0, 1.0, -1.0])])
case("EyeLike_k", [h.make_node("EyeLike", ["X"], ["Y"], k=1)], [vi("X", F, [3, 4])], y_out([3, 4]), {"X": f32(3, 4)})
case("Range", [h.make_node("Range", ["s", "l", "d"], ["Y"])], [], y_out([4]), {},
     inits=[h.make_tensor("s", F, [], [1.0]), h.make_tensor("l", F, [], [3.0]), h.make_tensor("d", F, [], [0.5])])
case("Einsum_bmm", [h.make_node("Einsum", ["X", "Z"], ["Y"], equation="bij,bjk->bik")],
     [vi("X", F, [2, 3, 4]), vi("Z", F, [2, 4, 5])], y_out([2, 3, 5]), {"X": X, "Z": f32(2, 4, 5)})
case("Gemm_trans_alpha_beta", [h.make_node("Gemm", ["A", "B", "C"], ["Y"], transA=1, transB=1, alpha=0.5, beta=2.0)],
     [vi("A", F, [4, 3])], y_out([3, 5]), {"A": f32(4, 3)},
     inits=[h.make_tensor("B", F, [5, 4], f32(5, 4).ravel()), h.make_tensor("C", F, [5], f32(5).ravel())])
case("Cast_float_to_int64", [h.make_node("Cast", ["X"], ["Y"], to=I64)], x_in, [vi("Y", I64, [2, 3, 4])], {"X": X * 3})
case("Sigmoid", [h.make_node("Sigmoid", ["X"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X})
case("Erf", [h.make_node("Erf", ["X"], ["Y"])], x_in, y_out([2, 3, 4]), {"X": X})


# --- focused Gemm / Softmax variants ---
for ta in (0, 1):
    for tb in (0, 1):
        for alpha, beta in ((1.0, 1.0), (0.5, 1.0), (1.0, 2.0)):
            for cshape in ("none", "N", "1N", "M1", "MN", "scalar"):
                M, K, N = 3, 4, 5
                A = f32(*((K, M) if ta else (M, K)))
                Bm = f32(*((N, K) if tb else (K, N)))
                inits = [h.make_tensor("B", F, list(Bm.shape), Bm.ravel())]
                ins = ["A", "B"]
                if cshape != "none":
                    shp = {"N": [N], "1N": [1, N], "M1": [M, 1], "MN": [M, N], "scalar": []}[cshape]
                    C = f32(*shp) if shp else np.array(0.7, dtype=np.float32)
                    inits.append(h.make_tensor("C", F, shp, np.atleast_1d(C).ravel()))
                    ins.append("C")
                case(f"Gemm_tA{ta}_tB{tb}_a{alpha}_b{beta}_C{cshape}",
                     [h.make_node("Gemm", ins, ["Y"], transA=ta, transB=tb, alpha=alpha, beta=beta)],
                     [vi("A", F, list(A.shape))], y_out([M, N]), {"A": A}, inits=inits)
for opset in (11, 13):
    for axis in (None, 0, 1, 2, -1):
        kw = {} if axis is None else {"axis": axis}
        case(f"Softmax_op{opset}_axis{axis}", [h.make_node("Softmax", ["X"], ["Y"], **kw)], x_in, y_out([2, 3, 4]), {"X": X}, opset=opset)
    case(f"Softmax_op{opset}_rank2_default", [h.make_node("Softmax", ["X2"], ["Y"])], [vi("X2", F, [3, 4])], y_out([3, 4]),
         {"X2": f32(3, 4)}, opset=opset)

def run_sofie(m, name, feeds, out_names):
    path = f"/tmp/sofie_{name}.onnx"
    onnx.save(m, path)
    parser = SOFIE.RModelParser_ONNX()
    rmodel = parser.Parse(path)
    rmodel.Generate()
    hdr = f"/tmp/sofie_{name}.hxx"
    rmodel.OutputGenerated(hdr)
    ROOT.gInterpreter.Declare(f'#include "{hdr}"')
    ns = getattr(ROOT, f"TMVA_SOFIE_sofie_{name}")
    session = ns.Session(f"/tmp/sofie_{name}.dat") if os.path.exists(f"/tmp/sofie_{name}.dat") else ns.Session()
    args = [np.ascontiguousarray(feeds[i.name]) for i in m.graph.input]
    res = session.infer(*args)
    if len(out_names) == 1:
        return [np.asarray(res)]
    return [np.asarray(r) for r in res]


results = []
FILTER = os.environ.get("FILTER", "")
for name, nodes, inputs, outputs, feeds, inits, opset in CASES:
    if FILTER and not __import__("re").search(FILTER, name):
        continue
    try:
        m = model(f"sofie_{name}", nodes, inputs, outputs, inits, opset)
    except Exception as e:
        results.append((name, "BADMODEL", str(e)[:200]))
        continue
    try:
        ref = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)
    except Exception as e:
        results.append((name, "ORTFAIL", str(e)[:200]))
        continue
    try:
        got = run_sofie(m, name, feeds, [o.name for o in outputs])
    except Exception as e:
        results.append((name, "SOFIEFAIL", (str(e).splitlines() or [""])[-1][:300]))
        continue
    msgs = []
    for o, r, g in zip(outputs, ref, got):
        g = g.ravel()
        r = np.asarray(r)
        if g.size != r.size:
            msgs.append(f"{o.name}: size {g.size} vs ref {r.size}")
        elif not np.allclose(g.astype(np.float64), r.ravel().astype(np.float64), rtol=1e-4, atol=1e-5):
            d = np.abs(g.astype(np.float64) - r.ravel().astype(np.float64))
            msgs.append(f"{o.name}: max abs diff {d.max():.3g}; sofie[:6]={np.round(g[:6], 4).tolist()} ref[:6]={np.round(r.ravel()[:6], 4).tolist()}")
    results.append((name, "MISMATCH" if msgs else "OK", "; ".join(msgs)))

print("ROOT", ROOT.gROOT.GetVersion())
for name, status, msg in results:
    print(f"RESULT {status:9s} {name} {msg}")
