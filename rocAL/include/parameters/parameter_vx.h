/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <VX/vx_compatibility.h>
#include <VX/vx_types.h>

#include <vector>

#include "parameter_factory.h"

template <typename T>
class ParameterVX {
   public:
    ParameterVX(unsigned ovx_param_idx, T default_range_start, T default_range_end) : OVX_PARAM_IDX(ovx_param_idx),
                                                                                      _DEFAULT_RANGE_START(default_range_start),
                                                                                      _DEFAULT_RANGE_END(default_range_end) {
        _param = ParameterFactory::instance()->create_uniform_rand_param<T>(_DEFAULT_RANGE_START,
                                                                            _DEFAULT_RANGE_END);
    }
    ParameterVX(T default_range_start, T default_range_end) : _DEFAULT_RANGE_START(default_range_start),
                                                              _DEFAULT_RANGE_END(default_range_end) {
        _param = ParameterFactory::instance()->create_uniform_rand_param<T>(_DEFAULT_RANGE_START,
                                                                            _DEFAULT_RANGE_END);
    }
    void create(vx_node node) {
        vx_status status;
        auto ref = vxGetParameterByIndex(node, OVX_PARAM_IDX);
        if ((status = vxQueryParameter(ref, VX_PARAMETER_ATTRIBUTE_REF, &_scalar, sizeof(vx_scalar))) != VX_SUCCESS ||
            (status = vxGetStatus((vx_reference)node)) != VX_SUCCESS)
            THROW("Getting vx scalar from the vx node failed" + TOSTR(status));
        if ((status = vxReadScalarValue(_scalar, &_val)) != VX_SUCCESS)
            THROW("Reading vx scalar failed" + TOSTR(status));
    }
    void create_array(std::shared_ptr<Graph> graph, vx_enum data_type, unsigned batch_size) {
        _batch_size = batch_size;
        _param->create_array(_batch_size);
        _array = vxCreateArray(vxGetContext((vx_reference)graph->get()), data_type, _batch_size);
        auto status = vxAddArrayItems(_array, _batch_size, get_array().data(), sizeof(T));
        if (status != 0)
            THROW(" vxAddArrayItems failed in create_array (ParameterVX): " + TOSTR(status))
        update_array();
    }

    void create_tensor(std::shared_ptr<Graph> graph, vx_enum data_type, unsigned batch_size) {
        vx_size dims[1] = { batch_size };
        _batch_size = batch_size;
        vx_size stride_output[1] = {sizeof(get_array()[0])};
        _param->create_array(_batch_size);
        _tensor = vxCreateTensorFromHandle(vxGetContext((vx_reference)graph->get()), 1, dims, data_type, 0, stride_output, get_array().data(), VX_MEMORY_TYPE_HOST);
        update_tensor();
    }

    void set_tensor(vx_tensor external_source_tensor) {
        _tensor = external_source_tensor;  
        if(_tensor == nullptr)
            std::cerr << "\n Tensor is a null PTR"; 
    }
    
    void set_param(Parameter<T>* param) {
        if (!param)
            return;

        ParameterFactory::instance()->destroy_param(_param);
        _param = param;
    }
    void set_param(T val) {
        ParameterFactory::instance()->destroy_param(_param);
        _param = ParameterFactory::instance()->create_single_value_param(val);
    }


    T default_value() {
        return _param->default_value();
    }
    vx_array default_array() {
        return _array;
    }
    vx_tensor default_tensor() {
        return _tensor;
    }
    vx_scalar default_scalar(std::shared_ptr<Graph> _graph, vx_enum data_type) {
        _scalar = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), data_type, &_val);
        return _scalar;
    }
    T get() {
        return _val;
    }
    void update() {
        vx_status status;

        T val = _param->get();

        if (_val == val)
            return;

        if ((status = vxWriteScalarValue(_scalar, &val)) != VX_SUCCESS)
            WRN("Updating vx scalar failed")
    }

    void update_array() {
        vx_status status;
        status = vxCopyArrayRange((vx_array)_array, 0, _batch_size, sizeof(T), get_array().data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != 0)
            THROW(" vxCopyArrayRange failed in update_array (ParameterVX): " + TOSTR(status))
    }

    void update_tensor() {
        vx_status status;
        std::cerr << "\n paramater_vx.h - update_tensor - get_array()[0]" << get_array()[0];
        std::cerr << "\n sizeof(get_array()[0]) - " << sizeof(get_array()[0]);
        vx_size stride_output[1] = {sizeof(get_array()[0])};
        status = vxCopyTensorPatch((vx_tensor)_tensor, 1, nullptr, nullptr, stride_output, get_array().data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
            THROW("ERROR: vxCopyArrayRange failed in update_tensor (ParameterVX)" + TOSTR(status));
    }
// Remove this function later
    // void update_tensor() {
    //     vx_status status;
    //     // for (uint i = 0; i < _batch_size; i++) { 
    //     //     _arrVal[i] = renew();
    //     // }
    //     vx_size stride_output[1] = {sizeof(float)};
    //     vx_size output_dims[1];
    // }
    T renew() {
        _param->renew();
        return _param->get();
    }

    ~ParameterVX() {
        if(_tensor)
            vxReleaseTensor((vx_tensor*)&_tensor);
    }

    std::vector<T> get_array() {
        return _param->get_array();
    }

    std::vector<T> get_tensor() {
        return _param->get_tensor();
    }

   private:
    vx_scalar _scalar;
    vx_array _array = nullptr;
    vx_tensor _tensor = nullptr;
    Parameter<T>* _param;
    T _val;
    unsigned _batch_size;
    unsigned OVX_PARAM_IDX;
    const T _DEFAULT_RANGE_START;
    const T _DEFAULT_RANGE_END;
};
