$	y#??Me@?????qr@?W?\??!r5?+??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'r5?+??@??r-Z =@1?,z?B?|@I?NZ2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?W?\??]?E?~??1?mO???^?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!Q??9???1?&?|?g?I?,?뇠?r11*	??????9@2T
Iterator::Root::ParallelMapV2F%u???!     ?I@)F%u???1     ?I@:Preprocessing2E
Iterator::Roota??+e??!     8X@)???????1     ?F@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!      	@)-C??6J?1      	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIH~?dX?"@Q7Po??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(?\?l#@????&?0@!??r-Z =@	!       "$	?_?G9Rc@r[??}?p@?mO???^?!?,z?B?|@*	!       2	!       :	?(^em?@?&?^,%@!?NZ2@B	!       J	!       R	!       Z	!       b	!       JGPUb qH~?dX?"@y7Po??V@