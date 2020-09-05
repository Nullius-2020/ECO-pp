import paddle
import paddle.fluid as nn
import paddle.fluid as fluid

LAYER_BUILDER_DICT=dict()

def parse_expr(expr):
    parts = expr.split('<=')
    return parts[0].split(','), parts[1], parts[2].split(',')


def get_basic_layer(info, channels=None, conv_bias=False, num_segments=4):
    id = info['id']

    attr = info['attrs'] if 'attrs' in info else dict()
    if 'kernel_d' in attr.keys():
        if isinstance(attr["kernel_d"], str):
            div_num = int(attr["kernel_d"].split("/")[-1])
            attr['kernel_d'] = int(num_segments / div_num)

    out, op, in_vars = parse_expr(info['expr'])
    assert(len(out) == 1)
    assert(len(in_vars) == 1)

    mod, out_channel= LAYER_BUILDER_DICT[op](attr, channels, conv_bias,in_vars[0])
    return id, out[0], mod, out_channel, in_vars[0]


def build_conv(attr, channels=None, conv_bias=False,conv=None):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_h'], attr['kernel_w'])
    if 'pad' in attr or 'pad_w' in attr and 'pad_h' in attr:
        padding = attr['pad'] if 'pad' in attr else (attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if 'stride' in attr or 'stride_w' in attr and 'stride_h' in attr:
        stride = attr['stride'] if 'stride' in attr else (attr['stride_h'], attr['stride_w'])
    else:
        stride = 1
    conv =fluid.dygraph.Conv2D(num_channels=channels,num_filters=out_channels,filter_size=ks, stride=stride, padding=padding,bias_attr=conv_bias)
    return conv, out_channels


def build_pooling(attr, channels=None, conv_bias=False,conv=None):
    method = attr['mode']
    pad = attr['pad'] if 'pad' in attr else 0
    if method == 'max':
        pool =  paddle.fluid.dygraph.Pool2D(pool_size=attr['kernel_size'],pool_type='max', pool_stride=attr['stride'], pool_padding=pad,
                            ceil_mode=True) # all Caffe pooling use ceil model
    elif method == 'ave':
        pool = paddle.fluid.dygraph.Pool2D(pool_size=attr['kernel_size'], pool_type='avg',pool_stride=attr['stride'], pool_padding=pad,
                            ceil_mode=True)  # all Caffe pooling use ceil model
    else:
        raise ValueError("Unknown pooling method: {}".format(method))

    return pool, channels


def build_relu(attr, channels=None, conv_bias=False,conv=None):
    return lambda x:paddle.fluid.layers.relu(x), channels


def build_bn(attr, channels=None, conv_bias=False,conv=None):
    return paddle.fluid.dygraph.BatchNorm(channels,momentum=0.1), channels
    
def build_linear(attr, channels=None, conv_bias=False,conv=None):
    return paddle.fluid.dygraph.Linear(channels, attr['num_output']), channels



def build_dropout(attr, channels=None, conv_bias=False,conv=None):
    p=attr['dropout_ratio']
    return lambda x:paddle.fluid.layers.dropout(x,p), channels

def build_conv3d(attr, channels=None, conv_bias=False,conv=None):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
    if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
        padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
        stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
    else:
        stride = 1
    conv = paddle.fluid.dygraph.Conv3D(channels,num_filters=out_channels, filter_size=ks, groups=1,stride=stride, padding=padding, bias_attr=conv_bias)
    return conv, out_channels

def build_pooling3d(attr, channels=None, conv_bias=False,conv=None):
    method = attr['mode']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
    if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
        padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
        stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
    else:
        stride = 1
    if method == 'max':
        pool =lambda x: fluid.layers.pool3d(input=x,pool_size=ks,pool_stride=stride,ceil_mode=True,pool_padding=padding,pool_type='max')
    elif method == 'ave':
        pool = lambda x:fluid.layers.pool3d(input=x,pool_size=ks,pool_stride=stride,ceil_mode=True,pool_padding=padding,pool_type='avg')
    else:
        raise ValueError("Unknown pooling method: {}".format(method))
    return pool, channels

def build_bn3d(attr, channels=None, conv_bias=False,conv=None):
    return paddle.fluid.dygraph.BatchNorm(channels,momentum=0.1), channels


LAYER_BUILDER_DICT['Convolution'] = build_conv

LAYER_BUILDER_DICT['Pooling'] = build_pooling

LAYER_BUILDER_DICT['ReLU'] = build_relu

LAYER_BUILDER_DICT['Dropout'] = build_dropout

LAYER_BUILDER_DICT['BN'] = build_bn

LAYER_BUILDER_DICT['InnerProduct'] = build_linear

LAYER_BUILDER_DICT['Conv3d'] = build_conv3d

LAYER_BUILDER_DICT['Pooling3d'] = build_pooling3d

LAYER_BUILDER_DICT['BN3d'] = build_bn3d

