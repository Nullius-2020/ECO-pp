
import paddle.fluid as fluid
import paddle.fluid as tensor
import numpy as np
from .layer_factory import get_basic_layer, parse_expr
import yaml


class ECOfull(fluid.dygraph.Layer):
    def __init__(self, model_path='tf_model_zoo/ECOfull/ECOfull.yaml', num_classes=101,
                       num_segments=4, pretrained_parts='both'):

        super(ECOfull, self).__init__()

        self.num_segments = num_segments

        self.pretrained_parts = pretrained_parts

        self.num_classes=num_classes

        manifest = yaml.load(open(model_path),Loader=yaml.FullLoader)

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat' and op != 'Eltwise':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True if op == 'Conv3d' else True, num_segments=num_segments)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            elif op == 'Concat':
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = self._channel_dict[in_var[0]]
                self._channel_dict[out_var[0]] = channel


    def forward(self, inputs):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = inputs
        out=data_dict[self._op_list[0][-1]]
        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook
        for op in self._op_list:

            if op[1] != 'Concat' and op[1] != 'InnerProduct' and op[1] != 'Eltwise':
                # first 3d conv layer judge, the last 2d conv layer's output must be transpose from 4d to 5d
                if op[0] == 'res3a_2' or op[0] == 'global_pool2D_reshape_consensus':
                    if  op[0] == 'global_pool2D_reshape_consensus':
                        layer_output = data_dict[op[-1]]
                        layer_output=fluid.layers.reshape(layer_output,(-1, self.num_segments)+tuple(layer_output.shape[1:]))
                        layer_transpose_output = fluid.layers.transpose(layer_output,perm=[0,2,1,3,4])
                        data_dict[op[2]] = getattr(self, op[0])(layer_transpose_output)
                    else:
                        layer_output = data_dict[op[-1]]
                        layer_output=fluid.layers.reshape(layer_output,(-1, self.num_segments)+tuple(layer_output.shape[1:]))
                        layer_transpose_output = fluid.layers.transpose(layer_output, perm=[0,2,1,3,4])
                        data_dict[op[2]] = getattr(self, op[0])(layer_transpose_output)

                else:
                    if op[0] == 'ReLu':
                        data_dict[op[2]]=fluid.layers.relu( data_dict[op[2]])
                    elif op[0]=='Pooling':
                        data_dict[op[2]]=fluid.dygraph.Pool2D(data_dict[op[2]])
                    else:
                        data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                x = fluid.layers.reshape(x,(x.shape[0],-1))
                data_dict[op[2]] = getattr(self, op[0])(x)
            elif op[1] == 'Eltwise':
                try:
                    data_dict[op[2]] = fluid.layers.elementwise_add(data_dict[op[-1][0]],data_dict[op[-1][1]],1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].shape)
                    raise
            else:
                try:
                    temp=tuple(data_dict[x] for x in op[-1])
                    data_dict[op[2]]=fluid.layers.concat(temp, 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].shape)
                    raise
        return data_dict[self._op_list[-1][2]]
