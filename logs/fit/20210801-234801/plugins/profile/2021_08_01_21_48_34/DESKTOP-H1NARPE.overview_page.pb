?$	??V???^@a????~j@+l? [??!횐֘?v@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'횐֘?v@?G?3?5@1?????s@I?h???_<@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails +l? [??1rQ-"??[?I?>?-W???r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?ó???1rQ-"??[?I????4s??r11*	ffffffK@2T
Iterator::Root::ParallelMapV2??(\?¥?! ?.??cS@)??(\?¥?1 ?.??cS@:Preprocessing2E
Iterator::Root???S㥫?!\?w???X@)Zd;?O???1??#EC?4@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!)??[??)-C??6J?1)??[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?
??[+@Q??????U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	_
?]'@??? ?a(@!?G?3?5@	!       "$	MA~6r?Z@?>?$?f@rQ-"??[?!?????s@*	!       2	!       :$	u_???#@?/?>?0@?>?-W???!?h???_<@B	!       J	!       R	!       Z	!       b	!       JGPUb q?
??[+@y??????U@?"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_4_grad/Conv2DBackpropFilterConv2DBackpropFilter?Ǣ?????!?Ǣ?????0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_5_grad/Conv2DBackpropFilterConv2DBackpropFilter???kw??!C5??v~??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_6_grad/Conv2DBackpropFilterConv2DBackpropFilter÷[8q??!$?8?7??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_7_grad/Conv2DBackpropFilterConv2DBackpropFilterȨ?(p??!??f?w??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_4_grad/Conv2DBackpropInputConv2DBackpropInput?(lIIA??!?w2?????0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_7_grad/Conv2DBackpropInputConv2DBackpropInput?H2?h2??!??x?	???0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_6_grad/Conv2DBackpropInputConv2DBackpropInput??????!?D/z&??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_5_grad/Conv2DBackpropInputConv2DBackpropInput0 o?d???!?(????0"g
Kmodel/conv_lstm1d_2/while/body/_557/model/conv_lstm1d_2/while/convolution_7Conv2D??G????!9??jQ??"g
Kmodel/conv_lstm1d_2/while/body/_557/model/conv_lstm1d_2/while/convolution_5Conv2D8??^{Ѩ?!̿??????Q      Y@Y?lVъ??az&S]??X@q?ٲ??W@ypG???B?"?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Maxwell)(: B 