$	z9쾣^U@??Zx?\e@-C??6z?!2?F]u@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'2?F]u@}?.PR,6@1?*?^?s@I???߃?@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails )??????T?d???1]?E?~e?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C??6z?1-C??6z?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?&?|??????V_]??1?'eRC[?r11*	fffff?A@2T
Iterator::Root::ParallelMapV2?z6?>??!W?l|3?O@)?z6?>??1W?l|3?O@:Preprocessing2E
Iterator::Root?5?;Nѡ?!?
?:MX@)??@??ǈ?1aT??A?@@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????MbP?!x????X@)????MbP?1x????X@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?뒭~?@QH?&?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?`?B@,?I/?$&@!}?.PR,6@	!       "$	?Q*ቺS@?8I?O?c@?'eRC[?!?*?^?s@*	!       2	!       :	???߃??????߃???!???߃?@B	!       J	!       R	!       Z	!       b	!       JGPUb q?뒭~?@yH?&?W@