$	?q??9c@4?|??ip@??D????!? x|v|@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'? x|v|@?/K;5{8@1͒ 5??x@I1?Z{??B@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails N??????1rQ-"??k?I??ʆ5???r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??D????1?h㈵?d?I??25	???r12*	43333sA@2T
Iterator::Root::ParallelMapV2?:pΈ??!???9e?I@)?:pΈ??1???9e?I@:Preprocessing2E
Iterator::Rootz6?>W[??!?v?p?HX@)vq?-??1?!??עF@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????MbP?!i%1?1?@)????MbP?1i%1?1?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???`?+@Q>????U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	q??|#R @?????D,@!?/K;5{8@	!       "$	9?j?3c`@E?]#bl@?h㈵?d?!͒ 5??x@*	!       2	!       :$	?M+?@?)@R?Ś[j5@??25	???!1?Z{??B@B	!       J	!       R	!       Z	!       b	!       JGPUb q???`?+@y>????U@