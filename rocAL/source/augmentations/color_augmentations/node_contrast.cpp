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

#include <vx_ext_rpp.h>
#include "node_contrast.h"
#include "exception.h"

ContrastNode::ContrastNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                        _factor(CONTRAST_FACTOR_RANGE[0], CONTRAST_FACTOR_RANGE[1]),
                                                                                                        _center(CONTRAST_CENTER_RANGE[0], CONTRAST_CENTER_RANGE[1]) {}

void ContrastNode::create_node() {
    if (_node)
        return;

    if(_tensor_factor->info().is_external_source() == false) { // new
        _factor.create_tensor(_graph, VX_TYPE_FLOAT32,  _tensor_factor->info().dims()[0]);
    } else {
        _factor.set_tensor(_tensor_factor->handle());
    }
    if(_tensor_center->info().is_external_source() == false) { // new
        _center.create_tensor(_graph, VX_TYPE_FLOAT32,  _tensor_center->info().dims()[0]);
    } else {
        _center.set_tensor(_tensor_center->handle());
    }
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppContrast(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _factor.default_tensor(), _center.default_tensor(), input_layout_vx, output_layout_vx,roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the contrast (vxExtRppContrast) node failed: " + TOSTR(status))
}

void ContrastNode::init(float contrast_factor, float contrast_center) {
    _factor.set_param(contrast_factor);
    _center.set_param(contrast_center);
}

void ContrastNode::init(Tensor *contrast_factor_tensor, Tensor *contrast_center_tensor) {
    _tensor_factor = contrast_factor_tensor;
    _tensor_center = contrast_center_tensor;
    if(_tensor_factor->info().is_external_source() == false) { // new
        _factor.set_param(core(std::get<FloatParam*>(contrast_factor_tensor->get_param())));
    }
    if(_tensor_center->info().is_external_source() == false) { // new
        _center.set_param(core(std::get<FloatParam*>(contrast_center_tensor->get_param())));
    }
}

void ContrastNode::update_node() {
    if(_tensor_factor->info().is_external_source() == false) // new
        _factor.update_tensor();
    if(_tensor_center->info().is_external_source() == false) // new
        _center.update_tensor();
}