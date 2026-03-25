import * as air from '../source/air.js';
import assert from 'assert/strict';

const metadata = {
    type(name) {
        return { name };
    }
};

const context = {
    metadata,
    weights: null
};

const tensorDesc = (dimensions = [1], dtype = 1, attr = {}) => {
    return {
        shape: { dim: dimensions },
        dtype,
        layout: 'ND',
        attr
    };
};

const tensorDef = (data, desc = tensorDesc()) => {
    return {
        data: new Uint8Array(data),
        desc
    };
};

const attribute = (value, entry) => {
    return { value, ...entry };
};

const graph = (ops, outputs = []) => {
    return {
        name: 'main',
        op: ops,
        output: outputs
    };
};

const testUnknownAttributeFallback = () => {
    const value = air.Utility.attributeValue({
        value: 'mystery',
        payload: {
            nested: attribute('s', { s: new Uint8Array([65, 66]) })
        }
    }, context);
    assert.deepEqual(value, { payload: { nested: { s: 'AB' } } });
};

const testConstGraphOutputInitializer = () => {
    const model = new air.Model(metadata, {
        version: 1,
        graph: [
            graph([
                {
                    name: 'const_weight',
                    type: 'Const',
                    attr: {
                        value: attribute('t', { t: tensorDef([1, 2, 3, 4]) })
                    }
                }
            ], ['const_weight:0'])
        ]
    });
    const value = model.modules[0].outputs[0].value[0];
    assert.ok(value.initializer);
    assert.equal(value.initializer.values.length, 4);
    assert.equal(value.type.toString(), 'float32[1]');
};

const testTensorAttributesUseData = () => {
    const node = new air.Graph(context, graph([
        {
            name: 'node',
            type: 'Custom',
            attr: {
                tensor_attr: attribute('t', {
                    t: tensorDef([9, 8, 7, 6], tensorDesc([4]))
                }),
                tensor_list: attribute('list', {
                    list: {
                        t: [tensorDef([6, 7, 8, 9], tensorDesc([4]))]
                    }
                })
            }
        }
    ])).nodes[0];
    assert.equal(node.attributes[0].value.values.length, 4);
    assert.equal(node.attributes[1].value[0].values.length, 4);
};

const testUtf8Strings = () => {
    const text = new Uint8Array([0xe4, 0xbd, 0xa0, 0xe5, 0xa5, 0xbd]);
    const node = new air.Graph(context, graph([
        {
            name: 'node',
            type: 'Custom',
            attr: {
                labels: attribute('list', {
                    list: {
                        s: [text]
                    }
                })
            }
        }
    ])).nodes[0];
    assert.equal(node.attributes[0].value, '你好');
    assert.equal(air.Utility.attributeValue(attribute('s', { s: text }), context), '你好');
};

const testNodeStringAttribute = () => {
    const text = new Uint8Array([0xe4, 0xbd, 0xa0, 0xe5, 0xa5, 0xbd]);
    const node = new air.Graph(context, graph([
        {
            name: 'node',
            type: 'Custom',
            attr: {
                label: attribute('s', { s: text })
            }
        }
    ])).nodes[0];
    assert.equal(node.attributes[0].value, '你好');
};

const testControlCharactersFallback = () => {
    const bytes = new Uint8Array([0x41, 0x00, 0x42]);
    const value = air.Utility.decodeBytes(bytes);
    assert.ok(value instanceof Uint8Array);
    assert.deepEqual(Array.from(value), [0x41, 0x00, 0x42]);
};

const testInvalidUtf8Fallback = () => {
    const bytes = new Uint8Array([0xc3, 0x28]);
    const value = air.Utility.decodeBytes(bytes);
    assert.ok(value instanceof Uint8Array);
    assert.deepEqual(Array.from(value), [0xc3, 0x28]);
};

const testFormatDisplay = () => {
    const descriptor = air.Utility.tensorDescriptor(tensorDesc([1], 1, {
        format: attribute('s', { s: new Uint8Array([78, 68]) }),
        format_for_int: attribute('i', { i: 2 })
    }), context);
    assert.equal(descriptor.attr.format_for_int, 'ND (2)');
};

const tests = [
    ['unknown attribute fallback', testUnknownAttributeFallback],
    ['const graph output initializer', testConstGraphOutputInitializer],
    ['tensor attributes use data', testTensorAttributesUseData],
    ['utf-8 string decoding', testUtf8Strings],
    ['node string attribute decoding', testNodeStringAttribute],
    ['control character fallback', testControlCharactersFallback],
    ['invalid utf-8 fallback', testInvalidUtf8Fallback],
    ['format display', testFormatDisplay]
];

for (const [name, test] of tests) {
    test();
    process.stdout.write(`air: ${name}\n`);
}
