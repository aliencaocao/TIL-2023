_base_ = [
    '../_base_/base_static.py',
    '../../_base_/backends/tensorrt.py'
]

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(shape=[1, 3, 1280, 720])))
    ])
