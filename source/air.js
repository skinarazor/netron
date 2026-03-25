// Experimental

import './base.js';
import * as protobuf from './protobuf.js';

const air = {};

air.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (!stream || stream.length === 0) {
            return null;
        }
        const position = stream.position;
        const buffer = stream.peek(Math.min(stream.length, 4));
        const signature = Array.from(buffer).map((c) => String.fromCharCode(c)).join('');
        if (signature === 'IMOD' || signature === 'PICO') {
            return null;
        }
        const module = await context.require('./om-proto');
        const proto = module.ge.proto;
        try {
            const reader = protobuf.BinaryReader.open(stream);
            const model = proto.ModelDef.decode(reader);
            if (Array.isArray(model.graph) && model.graph.length > 0) {
                return context.set('air', model);
            }
        } catch {
            // continue regardless of error
        } finally {
            if (stream.position !== position) {
                stream.seek(position);
            }
        }
        return null;
    }

    async open(context) {
        const metadata = await context.metadata('om-metadata.json');
        return new air.Model(metadata, context.value);
    }
};

air.Model = class {

    constructor(metadata, model) {
        this.format = 'Ascend AIR';
        this.version = model.version ? model.version.toString() : model.custom_version || '';
        const context = {
            metadata,
            weights: null
        };
        this.modules = model.graph.map((graph) => new air.Graph(context, graph));
    }
};

air.Graph = class {

    constructor(context, graph) {
        this.name = graph.name || '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (values.has(name)) {
                const value = values.get(name);
                if (type) {
                    if (value.type === null) {
                        value.type = type;
                    } else if (!type.equals(value.type)) {
                        value.type = air.Utility.mergeType(value.type, type);
                    }
                }
                if (tensor) {
                    if (value.initializer === null) {
                        value.initializer = tensor;
                    } else if (tensor !== value.initializer) {
                        throw new air.Error(`Duplicate value '${name}'.`);
                    }
                }
            } else {
                values.set(name, new air.Value(name, type || null, tensor || null));
            }
            return values.get(name);
        };
        const tensors = new Map();
        const ops = [];
        const netOutputs = [];
        for (const op of graph.op) {
            if (op.type === 'Data' && op.output_desc) {
                for (let i = 0; i < op.output_desc.length; i++) {
                    const identifier = `${op.name}:${i}`;
                    const type = air.Utility.tensorType(op.output_desc[i]);
                    const name = i === 0 ? op.name : `${op.name}:${i}`;
                    const value = values.map(identifier, type);
                    this.inputs.push(new air.Argument(name, [value]));
                }
                continue;
            }
            if (op.type === 'Const' && op.attr && op.attr.value) {
                const initializer = air.Utility.createTensor(op.attr.value.t, context, op.name);
                tensors.set(op.name, initializer);
                continue;
            }
            if (op.type === 'NetOutput' && op.input) {
                for (const input of op.input) {
                    if (input !== '') {
                        netOutputs.push(input);
                    }
                }
                continue;
            }
            ops.push(op);
        }
        for (const op of ops) {
            const node = new air.Node(context, op, graph, values, tensors);
            this.nodes.push(node);
        }
        const outputs = Array.isArray(graph.output) && graph.output.length > 0 ? graph.output : netOutputs;
        for (let i = 0; i < outputs.length; i++) {
            const identifier = outputs[i];
            const tensor = air.Utility.tensor(identifier, tensors);
            const type = tensor ? tensor.type : null;
            const value = values.map(identifier, type, tensor);
            const name = i === 0 ? 'output' : `output${i}`;
            this.outputs.push(new air.Argument(name, [value]));
        }
    }
};

air.Node = class {

    constructor(context, op, graph, values, tensors) {
        this.name = op.name || '';
        this.type = context.metadata.type(op.type) || { name: op.type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        this.controlDependencies = [];
        this.device = null;
        if (op.input) {
            let index = 0;
            for (let i = 0; i < op.input.length; i++) {
                const input = op.input[i];
                if (input === '') {
                    continue;
                }
                const name = this.type.inputs && i < this.type.inputs.length ? this.type.inputs[i].name : `input${index === 0 ? '' : index}`;
                index++;
                const end = this.type.inputs && i < this.type.inputs.length && this.type.inputs[i].type && this.type.inputs[i].type === 'Tensor[]' ? op.input.length : i + 1;
                const list = [];
                for (let j = i; j < end; j++) {
                    const input = op.input[j];
                    if (input === '') {
                        continue;
                    }
                    const index = input.lastIndexOf(':');
                    const identifier = input.substring(0, index);
                    const src_index = input.substring(index + 1);
                    if (src_index === '-1') {
                        this.controlDependencies.push(values.map(name));
                        continue;
                    }
                    const type = air.Utility.tensorType(op.input_desc[j]);
                    const tensor = tensors.get(identifier);
                    const value = values.map(input, type, tensor);
                    list.push(value);
                }
                const argument = new air.Argument(name, list);
                this.inputs.push(argument);
                i = end - 1;
            }
        }
        if (op.output_desc) {
            for (let i = 0; i < op.output_desc.length; i++) {
                const identifier = `${this.name}:${i}`;
                const type = air.Utility.tensorType(op.output_desc[i]);
                const name = this.type.outputs && i < this.type.outputs.length ? this.type.outputs[i].name : `output${i === 0 ? '' : i}`;
                const value = values.map(identifier, type);
                const argument = new air.Argument(name, [value]);
                this.outputs.push(argument);
            }
        }
        for (const [name, obj] of Object.entries(op.attr || {})) {
            if (name === 'device') {
                this.device = obj;
                continue;
            }
            if (name === 'relu_flag' && obj.b) {
                const node = new air.Node(context, { type: 'ReLU' }, graph, obj);
                this.chain.push(node);
                continue;
            }
            let value = obj;
            let type = null;
            switch (obj.value) {
                case 'i': {
                    value = obj.i;
                    type = 'int64';
                    break;
                }
                case 'f': {
                    value = obj.f;
                    type = 'float32';
                    break;
                }
                case 'b': {
                    value = obj.b;
                    type = 'boolean';
                    break;
                }
                case 'bt': {
                    value = null;
                    if (obj.bt.length !== 0) {
                        type = 'tensor';
                        const shape = new air.TensorShape([obj.bt.length / 4]);
                        value = new air.Tensor('Constant', new air.TensorType('float32', shape), obj.bt);
                    }
                    break;
                }
                case 'dt': {
                    type = 'DataType';
                    value = air.Utility.dtype(Number(obj.dt));
                    break;
                }
                case 's': {
                    if (typeof obj.s === 'string') {
                        value = obj.s;
                    } else if (obj.s.every((c) => c >= 32 && c <= 128)) {
                        value = air.Utility.decodeText(obj.s);
                    } else {
                        value = obj.s;
                    }
                    type = 'string';
                    break;
                }
                case 'g': {
                    type = 'graph';
                    value = new air.Graph(context, obj.g);
                    break;
                }
                case 'func': {
                    type = 'function';
                    value = air.Utility.namedAttrs(obj.func, context);
                    break;
                }
                case 'td': {
                    type = 'tensor';
                    value = air.Utility.tensorDescriptor(obj.td, context);
                    break;
                }
                case 'list': {
                    const list = obj.list;
                    value = [];
                    if (list.s && list.s.length > 0) {
                        value = list.s.map((v) => String.fromCharCode.apply(null, new Uint16Array(v))).join(', ');
                        type = 'string[]';
                    } else if (list.b && list.b.length > 0) {
                        value = list.b;
                        type = 'boolean[]';
                    } else if (list.i && list.i.length > 0) {
                        value = list.i;
                        type = 'int64[]';
                    } else if (list.f && list.f.length > 0) {
                        value = list.f;
                        type = 'float32[]';
                    } else if (list.bt && list.bt.length > 0) {
                        value = list.bt;
                        type = 'byte[]';
                    } else if (list.td && list.td.length > 0) {
                        value = list.td.map((td) => air.Utility.tensorDescriptor(td, context));
                        type = 'tensor[]';
                    } else if (list.t && list.t.length > 0) {
                        value = list.t.map((tensor) => air.Utility.createTensor(tensor, context));
                        type = 'tensor[]';
                    } else if (list.g && list.g.length > 0) {
                        value = list.g.map((graph) => new air.Graph(context, graph));
                        type = 'graph[]';
                    } else if (list.na && list.na.length > 0) {
                        value = list.na.map((named) => air.Utility.namedAttrs(named, context));
                        type = 'function[]';
                    } else if (list.dt && list.dt.length > 0) {
                        value = list.dt.map((dt) => air.Utility.dtype(Number(dt)));
                        type = 'DataType[]';
                    } else if (list.type && list.type.length > 0) {
                        type = 'type[]';
                        value = list.type.map((type) => air.Node.enum2Dtype(type) || '?');
                    } else if (list.shape && list.shape.length > 0) {
                        type = 'shape[]';
                        value = list.shape.map((shape) => new air.TensorShape(shape));
                    } else {
                        type = 'list';
                        value = air.Utility.rawObject(list, context);
                    }
                    break;
                }
                case 'list_list_int': {
                    value = obj.list_list_int.list_list_i.map((list) => list.list_i);
                    type = 'int64[][]';
                    break;
                }
                case 'list_list_float': {
                    value = obj.list_list_float.list_list_f.map((list) => list.list_f);
                    type = 'float32[][]';
                    break;
                }
                case 't': {
                    type = 'tensor';
                    value = air.Utility.createTensor(obj.t, context);
                    break;
                }
                case undefined: {
                    value = null;
                    break;
                }
                default: {
                    type = obj.value || '?';
                    value = air.Utility.attributeValue(obj, context);
                    break;
                }
            }
            const attribute = new air.Argument(name, value, type);
            this.attributes.push(attribute);
        }
    }
};

air.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

air.Value = class {

    constructor(name, type, initializer = null) {
        if (typeof name !== 'string') {
            throw new air.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

air.Tensor = class {

    constructor(category, type, value, name = '', attributes = []) {
        this.category = category;
        this.name = name;
        this.type = type;
        this.values = value;
        this.attributes = attributes;
    }
};

air.TensorType = class {

    constructor(dataType, shape, denotation) {
        this.dataType = dataType;
        this.shape = shape;
        this.denotation = denotation;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

air.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
    }

    equals(obj) {
        if (obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions)) {
            if (this.dimensions.length === obj.dimensions.length &&
                obj.dimensions.every((value, index) => this.dimensions[index] === value)) {
                return true;
            }
            if (obj.dimensions.every((dim) => Number.isInteger(dim)) && this.dimensions.every((dim) => Number.isInteger(dim))) {
                const a = obj.dimensions.reduce((a, b) => a * b, 1);
                const b = this.dimensions.reduce((a, b) => a * b, 1);
                return a === b;
            }
        }
        return false;
    }

    toString() {
        if (this.dimensions && Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dim) => dim ? dim.toString() : '?').join(',')}]`;
        }
        return '';
    }
};

air.Utility = class {

    static dtype(value) {
        air.Utility._types = air.Utility._types || [
            'undefined', 'float32', 'float16', 'int8', 'uint8', 'int16', 'uint16', 'int32',
            'int64', 'uint32', 'uint64', 'boolean', 'float64', 'string', 'dual_sub_int8', 'dual_sub_uint8',
            'complex<float32>', 'complex<float64>', 'qint8', 'qint16', 'qint32', 'quint8', 'quint16', 'resource',
            'stringref', 'dual', 'variant', 'bfloat16', 'int4', 'uint1', 'int2', 'uint2'
        ];
        if (value >= air.Utility._types.length) {
            throw new air.Error(`Unsupported dtype '${value}'.`);
        }
        return air.Utility._types[value];
    }

    static tensorType(desc) {
        if (desc && desc.shape && Array.isArray(desc.shape.dim)) {
            const dataType = desc && desc.dtype ? air.Utility.dtype(desc.dtype) : '?';
            const shape = new air.TensorShape(desc.shape.dim);
            return new air.TensorType(dataType, shape, desc.layout);
        }
        return null;
    }

    static decodeText(value) {
        air.Utility._textDecoder = air.Utility._textDecoder || new TextDecoder('utf-8');
        return air.Utility._textDecoder.decode(value);
    }

    static tensorAttributes(desc, context) {
        if (!desc || !desc.attr || Object.keys(desc.attr).length === 0) {
            return [];
        }
        return Object.entries(desc.attr).map(([name, value]) => air.Utility.tensorAttribute(name, value, desc.attr, context));
    }

    static tensorAttribute(name, value, attributes, context) {
        const type = value && value.value ? value.value : null;
        if ((name === 'origin_format_for_int' || name === 'format_for_int') && value && value.value === 'i') {
            const intValue = typeof value.i.toNumber === 'function' ? value.i.toNumber() : Number(value.i);
            const sibling = name === 'origin_format_for_int' ? attributes.origin_format : attributes.format;
            let format = null;
            if (sibling && sibling.value === 's' && sibling.s && sibling.s.length > 0) {
                format = air.Utility.decodeBytes(sibling.s);
            }
            format = format || air.Utility.dataFormat(intValue);
            if (format) {
                return new air.Argument(name, `${format} (${intValue})`, 'format');
            }
        }
        return new air.Argument(name, air.Utility.attributeValue(value, context), type);
    }

    static dataFormat(value) {
        air.Utility._formats = air.Utility._formats || new Map([
            [0, 'NCHW'],
            [1, 'NHWC'],
            [2, 'ND'],
            [3, 'NC1HWC0'],
            [4, 'FRACTAL_Z'],
            [16, 'HWCN']
        ]);
        return air.Utility._formats.get(value) || null;
    }

    static mergeType(current, candidate) {
        if (!current) {
            return candidate || null;
        }
        if (!candidate || current.equals(candidate)) {
            return current;
        }
        if (current.dataType === '?' && candidate.dataType !== '?') {
            return candidate;
        }
        if (candidate.dataType === '?' && current.dataType !== '?') {
            return current;
        }
        if (!current.shape && candidate.shape) {
            return candidate;
        }
        if (current.shape && !candidate.shape) {
            return current;
        }
        if (current.shape && candidate.shape && current.shape.dimensions.length === candidate.shape.dimensions.length) {
            return current;
        }
        return current;
    }

    static tensorDescriptor(desc, context) {
        if (!desc) {
            return null;
        }
        const type = air.Utility.tensorType(desc);
        if (type && !desc.attr) {
            return type;
        }
        const value = {};
        if (type) {
            value.type = type;
        }
        if (desc.name) {
            value.name = desc.name;
        }
        if (desc.layout) {
            value.layout = desc.layout;
        }
        if (desc.shape && Array.isArray(desc.shape.dim)) {
            value.shape = new air.TensorShape(desc.shape.dim);
        }
        if (desc.has_out_attr !== undefined) {
            value.has_out_attr = desc.has_out_attr;
        }
        if (desc.attr && Object.keys(desc.attr).length > 0) {
            value.attr = Object.fromEntries(air.Utility.tensorAttributes(desc, context).map((attribute) => [attribute.name, attribute.value]));
        }
        return value;
    }

    static namedAttrs(named, context) {
        if (!named) {
            return null;
        }
        const value = {};
        if (named.name) {
            value.name = named.name;
        }
        if (named.attr && Object.keys(named.attr).length > 0) {
            value.attr = Object.fromEntries(Object.entries(named.attr).map(([name, attr]) => [name, air.Utility.attributeValue(attr, context)]));
        }
        return value;
    }

    static attributeValue(obj, context) {
        if (!obj || obj.value === undefined) {
            return null;
        }
        switch (obj.value) {
            case 'i':
                return obj.i;
            case 'f':
                return obj.f;
            case 'b':
                return obj.b;
            case 'bt':
                return obj.bt;
            case 'dt':
                return air.Utility.dtype(Number(obj.dt));
            case 's':
                return typeof obj.s === 'string' ? obj.s : air.Utility.decodeBytes(obj.s);
            case 'g':
                return new air.Graph(context, obj.g);
            case 'func':
                return air.Utility.namedAttrs(obj.func, context);
            case 'td':
                return air.Utility.tensorDescriptor(obj.td, context);
            case 't':
                return air.Utility.createTensor(obj.t, context);
            case 'list':
                return air.Utility.listValue(obj.list, context);
            case 'list_list_int':
                return obj.list_list_int.list_list_i.map((list) => list.list_i);
            case 'list_list_float':
                return obj.list_list_float.list_list_f.map((list) => list.list_f);
            default:
                return air.Utility.rawAttributeValue(obj, context);
        }
    }

    static listValue(list, context) {
        if (!list) {
            return [];
        }
        if (list.s && list.s.length > 0) {
            return list.s.map((value) => air.Utility.decodeBytes(value)).join(', ');
        }
        if (list.b && list.b.length > 0) {
            return list.b;
        }
        if (list.i && list.i.length > 0) {
            return list.i;
        }
        if (list.f && list.f.length > 0) {
            return list.f;
        }
        if (list.bt && list.bt.length > 0) {
            return list.bt;
        }
        if (list.td && list.td.length > 0) {
            return list.td.map((td) => air.Utility.tensorDescriptor(td, context));
        }
        if (list.t && list.t.length > 0) {
            return list.t.map((tensor) => air.Utility.createTensor(tensor, context));
        }
        if (list.g && list.g.length > 0) {
            return list.g.map((graph) => new air.Graph(context, graph));
        }
        if (list.na && list.na.length > 0) {
            return list.na.map((named) => air.Utility.namedAttrs(named, context));
        }
        if (list.dt && list.dt.length > 0) {
            return list.dt.map((dt) => air.Utility.dtype(Number(dt)));
        }
        if (list.type && list.type.length > 0) {
            return list.type.map((type) => air.Node.enum2Dtype(type) || '?');
        }
        if (list.shape && list.shape.length > 0) {
            return list.shape.map((shape) => new air.TensorShape(shape));
        }
        return air.Utility.rawObject(list, context);
    }

    static decodeBytes(value) {
        if (typeof value === 'string') {
            return value;
        }
        if (Array.isArray(value) || value instanceof Uint8Array) {
            const buffer = value instanceof Uint8Array ? value : new Uint8Array(value);
            if (buffer.every((c) => c >= 32 && c <= 128)) {
                return air.Utility.decodeText(buffer);
            }
        }
        return value;
    }

    static tensor(identifier, tensors) {
        const index = identifier.lastIndexOf(':');
        const name = index === -1 ? identifier : identifier.substring(0, index);
        return tensors.get(name) || null;
    }

    static tensorData(tensor, context) {
        const desc = tensor.desc;
        if (tensor.data.length !== 0) {
            return tensor.data;
        }
        if (context.weights === null) {
            return null;
        }
        if (desc.attr.merged_offset) {
            const offset = desc.attr.merged_offset.i.toNumber();
            return context.weights.slice(offset, offset + desc.weight_size.toNumber());
        }
        const offset = desc.data_offset.toNumber();
        return context.weights.slice(offset, offset + desc.weight_size.toNumber());
    }

    static createTensor(tensor, context, name = '') {
        return new air.Tensor(
            'Constant',
            air.Utility.tensorType(tensor.desc),
            air.Utility.tensorData(tensor, context),
            name,
            air.Utility.tensorAttributes(tensor.desc, context)
        );
    }

    static rawAttributeValue(obj, context) {
        return Object.fromEntries(Object.entries(obj)
            .filter(([name, value]) => name !== 'value' && value !== null && value !== undefined)
            .map(([name, value]) => [name, air.Utility.rawObject(value, context)]));
    }

    static rawObject(obj, context) {
        if (Array.isArray(obj)) {
            return obj.map((item) => air.Utility.rawObject(item, context));
        }
        if (!obj || typeof obj !== 'object') {
            return obj;
        }
        if (obj instanceof Uint8Array) {
            return air.Utility.decodeBytes(obj);
        }
        if (obj.value !== undefined) {
            return air.Utility.rawAttributeValue(obj, context);
        }
        return Object.fromEntries(Object.entries(obj)
            .filter(([, value]) => value !== null && value !== undefined)
            .map(([name, value]) => [name, air.Utility.rawObject(value, context)]));
    }
};

air.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Ascend AIR model.';
    }
};

export const ModelFactory = air.ModelFactory;
