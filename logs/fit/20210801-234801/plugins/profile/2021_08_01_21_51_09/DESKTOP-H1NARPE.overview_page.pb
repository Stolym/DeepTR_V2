?$	߿yqbH`@??U?v2l@?=\r?)??!\kF?kx@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'\kF?kx@??x[???@1??????t@I??0? y9@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?V?Sb??1rQ-"??[?I>??????r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?=\r?)??1? 3??O\?I??u6䟉?r11*	233333F@2T
Iterator::Root::ParallelMapV2??ǘ????!??n0E>R@)??ǘ????1??n0E>R@:Preprocessing2E
Iterator::Root46<?R??!??~G??X@)Ǻ?????1??@\?99@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!C?I .???)-C??6J?1C?I .???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI@o??1-@Q??YU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ݥ??%@[u6e*32@!??x[???@	!       "$	?Z?7??[@????h@rQ-"??[?!??????t@*	!       2	!       :$	?K{pd!@]"?c-@??u6䟉?!??0? y9@B	!       J	!       R	!       Z	!       b	!       JGPUb q@o??1-@y??YU@?"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_4_grad/Conv2DBackpropFilterConv2DBackpropFilter?<ބ???!?<ބ???0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_5_grad/Conv2DBackpropFilterConv2DBackpropFilter|eoD{??!>???{}??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_6_grad/Conv2DBackpropFilterConv2DBackpropFilter=???u??!?^??D8??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_7_grad/Conv2DBackpropFilterConv2DBackpropFilter???t??!h3??Gy??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_7_grad/Conv2DBackpropInputConv2DBackpropInput;?9 A??!/r?Fk??0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_5_grad/Conv2DBackpropInputConv2DBackpropInput8j??2??!v??????0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_4_grad/Conv2DBackpropInputConv2DBackpropInput??u????!?1.p???0"?
?gradient_tape/model/conv_lstm1d_2/while/model/conv_lstm1d_2/while_grad/body/_835/gradient_tape/model/conv_lstm1d_2/while/gradients/model/conv_lstm1d_2/while/convolution_6_grad/Conv2DBackpropInputConv2DBackpropInput?{Ȱk???!?@G?????0"g
Kmodel/conv_lstm1d_2/while/body/_557/model/conv_lstm1d_2/while/convolution_7Conv2D{?L?G???!?oX?O??"g
Kmodel/conv_lstm1d_2/while/body/_557/model/conv_lstm1d_2/while/convolution_5Conv2D?0????!?"	Y3???Q      Y@Y?lVъ??az&S]??X@q??3R?X@y]eSHB?"?
both?Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Maxwell)(: B 