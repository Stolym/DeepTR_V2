?$	??H7??b@5>Be?"p@??͎Tߵ?!?=$|??{@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?=$|??{@k}?Ж?4@1?]i)x@I$C??gpD@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??͎Tߵ?1rQ-"??[?Iw?Df.p??r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?`6????1rQ-"??[?I?3w????r11*	?????I@2Y
"Iterator::Root::ParallelMapV2::Zip?0?*??!??????D@)?0?*??1??????D@:Preprocessing2T
Iterator::Root::ParallelMapV2??ܵ?|??!i{?-	@@)??ܵ?|??1i{?-	@@:Preprocessing2E
Iterator::Root???_vO??!zKBi{M@)???S㥋?1?čOv?:@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIȮ????+@Q'J??@?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????s?@?;F?{#(@!k}?Ж?4@	!       "$	???y`@$6???k@rQ-"??[?!?]i)x@*	!       2	!       :$	I?????+@f?z?v7@w?Df.p??!$C??gpD@B	!       J	!       R	!       Z	!       b	!       JGPUb qȮ????+@y'J??@?U@?"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_4_grad/Conv2DBackpropFilterConv2DBackpropFilter7???????!7???????0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_5_grad/Conv2DBackpropFilterConv2DBackpropFilter????~??!?|?N???0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_7_grad/Conv2DBackpropFilterConv2DBackpropFilterF??m??!M?z??6??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_6_grad/Conv2DBackpropFilterConv2DBackpropFilter֙:ni??!?K??u??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_4_grad/Conv2DBackpropInputConv2DBackpropInput??Z>?ì?!r@>??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_5_grad/Conv2DBackpropInputConv2DBackpropInput??ʪ??!??p
????0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_7_grad/Conv2DBackpropInputConv2DBackpropInput??????!?????0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_6_grad/Conv2DBackpropInputConv2DBackpropInputUڴ?	???!c=?P=???0"g
Kmodel/conv_lstm1d_2/while/body/_557/model/conv_lstm1d_2/while/convolution_7Conv2Dx8?c???!9"
?4W??"g
Kmodel/conv_lstm1d_2/while/body/_557/model/conv_lstm1d_2/while/convolution_6Conv2D?Z?S????!?G????Q      Y@Y?lVъ??az&S]??X@q_??}?@y?????CB?"?

both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Maxwell)(: B 