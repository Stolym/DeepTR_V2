?$	?,T}f@c=???xs@t^c??ފ?!???!݀@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???!݀@v()?:@1?@ ??~@I7߈?Y?4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails t^c??ފ?H?Sȕz??1?t><K?a?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!c^G????1F?̱??n?Ib?qm???r11*	33333?B@2E
Iterator::Root㥛? ???!??C?eX@)?g??s???1X@?n?WL@:Preprocessing2T
Iterator::Root::ParallelMapV2K?=?U??!u?E]tD@)K?=?U??1u?E]tD@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!@?n?W@@)??H?}M?1@?n?W@@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?.?4!@Q>??j?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I?@?Tj!@mS??0%.@!v()?:@	!       "$	?zԼ?d@5A????q@?t><K?a?!?@ ??~@*	!       2	!       :	??T??@dY?CX'@!7߈?Y?4@B	!       J	!       R	!       Z	!       b	!       JGPUb q?.?4!@y>??j?V@?"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_6_grad/Conv2DBackpropFilterConv2DBackpropFilter
x?g????!
x?g????0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_7_grad/Conv2DBackpropFilterConv2DBackpropFilterep ;x???!8?SQ???0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_4_grad/Conv2DBackpropFilterConv2DBackpropFilter]??=???!?@b????0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_5_grad/Conv2DBackpropFilterConv2DBackpropFilter???ş???!?6? ???0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_7_grad/Conv2DBackpropInputConv2DBackpropInputh???*??!Ĝ֢???0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_4_grad/Conv2DBackpropInputConv2DBackpropInput?W?q???!??B?????0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_6_grad/Conv2DBackpropInputConv2DBackpropInputH	?????!?:????0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_5_grad/Conv2DBackpropInputConv2DBackpropInput??*???!??tv#Q??0"g
Kmodel/conv_lstm1d_1/while/body/_279/model/conv_lstm1d_1/while/convolution_7Conv2D?,TL??!???&R??"g
Kmodel/conv_lstm1d_1/while/body/_279/model/conv_lstm1d_1/while/convolution_4Conv2D??Q??ޟ?!ޝD?P??Q      Y@YY?JV????anjS???X@q???xX@yO!??:?"?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Maxwell)(: B 