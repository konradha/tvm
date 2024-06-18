import numpy as np
import tvm
from tvm import relay


if __name__ == '__main__':
    tvm.target.datatype.register("AI23Float", 150)

    tvm.target.datatype.register_op(
        tvm.target.datatype.create_lower_func(
            {
                (32, 32): "FloatToAI23Float32",
            }
        ),
        "Cast",
        "llvm",
        "float",
        "AI23Float",
    )

    

    tvm.target.datatype.register_op(
        tvm.target.datatype.create_lower_func({32: "AI23Float32Add"}),
        "Add",
        "llvm",
        "AI23Float",
    )
    tvm.target.datatype.register_op(
        tvm.target.datatype.create_lower_func({(32, 32): "AI23Float32ToFloat"}),
        "Cast",
        "llvm",
        "AI23Float",
        "float",
    )

    n = 100
    x = relay.var("x", shape=(n,), dtype="float32")
    y = relay.var("y", shape=(n,), dtype="float32")
    z = x + y

    
    program = relay.Function([x, y], z)
    module = tvm.IRModule.from_expr(program)
    np.random.seed(23)  # for reproducibility

    x_input = np.random.rand(n).astype("float32")
    y_input = np.random.rand(n).astype("float32")
# z_output = relay.create_executor(mod=module).evaluate()(x_input, y_input)
# print("z: {}".format(z_output))

    
    try:
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            x_myfloat = relay.cast(x, dtype="custom[AI23Float]32")
            y_myfloat = relay.cast(y, dtype="custom[AI23Float]32")
            z_myfloat = x_myfloat + y_myfloat
            z = relay.cast(z_myfloat, dtype="float32")
            program = relay.Function([x, y], z)

            module = tvm.IRModule.from_expr(program)
            module = relay.transform.InferType()(module)
            z_output_myfloat = relay.create_executor(
                "graph", mod=module).evaluate()(
                x_input, y_input)

            print("x:\t\t{}".format(x_input))
            print("y:\t\t{}".format(y_input))
            print("z (float18):\t{}".format(z_output_myfloat))
    except tvm.TVMError as e:
        print(e)
